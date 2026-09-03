"""Paired bootstrap comparison of two evaluation bundles.

Marginal bootstrap SEs in a single bundle describe cohort sampling: they answer
"how much would this model's AUPRC move on a different draw of variants". A
comparison between two models on the *same* variants is far less noisy, so the
delta must be bootstrapped with both models resampled together.

Resampling units follow the scoring protocol: Mendelian resamples
``match_group`` (the matched positive/negative cluster), SGE resamples rows
within each study.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

VARIANT_KEY = ["chrom", "pos", "ref", "alt"]
MENDELIAN_MIN_GROUPS = 5
SELF_CHECK_TOLERANCE = 1e-9


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-interpolated average precision, equivalent to scikit-learn's."""
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    positives = int(ordered.sum())
    if positives == 0 or positives == ordered.size:
        return float("nan")
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, ordered.size + 1)
    return float((cumulative[ordered == 1] / ranks[ordered == 1]).mean())


@dataclass(frozen=True)
class Cell:
    """One AUPRC cell: the row indices it scores and its aggregation group."""

    name: str
    group: str
    rows: np.ndarray


def _mendelian_cells(frame: pd.DataFrame) -> list[Cell]:
    cells = []
    for subset, part in frame.groupby("subset", sort=True):
        if part["match_group"].nunique() < MENDELIAN_MIN_GROUPS:
            continue
        cells.append(Cell(name=str(subset), group=str(subset), rows=part.index.to_numpy()))
    return cells


def _sge_cells(frame: pd.DataFrame) -> list[Cell]:
    cells = []
    for (study, subset), part in frame.groupby(["mavedb_urn", "subset"], sort=True):
        if subset not in ("missense_variant", "splicing"):
            continue
        labels = part["label"].to_numpy()
        if labels.all() or not labels.any():
            continue
        cells.append(
            Cell(name=f"{study}:{subset}", group=str(study), rows=part.index.to_numpy())
        )
    return cells


def _macro(values: dict[str, list[float]]) -> float:
    """Mean within each group, then mean across groups."""
    per_group = [np.nanmean(v) for v in values.values()]
    return float(np.nanmean(per_group))


def macro_auprc(cells: list[Cell], labels: np.ndarray, scores: np.ndarray) -> float:
    grouped: dict[str, list[float]] = {}
    for cell in cells:
        grouped.setdefault(cell.group, []).append(
            average_precision(labels[cell.rows], scores[cell.rows])
        )
    return _macro(grouped)


def paired_delta(
    frame: pd.DataFrame,
    cells: list[Cell],
    *,
    unit: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the candidate-minus-baseline macro AUPRC on shared units."""
    labels = frame["label"].to_numpy().astype(np.int8)
    baseline = frame["score_baseline"].to_numpy()
    candidate = frame["score_candidate"].to_numpy()
    point_baseline = macro_auprc(cells, labels, baseline)
    point_candidate = macro_auprc(cells, labels, candidate)

    units, unit_codes = np.unique(unit, return_inverse=True)
    rows_by_unit = [np.flatnonzero(unit_codes == code) for code in range(units.size)]
    cell_of_row = np.full(len(frame), -1, dtype=np.int32)
    for index, cell in enumerate(cells):
        cell_of_row[cell.rows] = index
    groups = [cell.group for cell in cells]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, units.size, units.size)
        rows = np.concatenate([rows_by_unit[u] for u in sampled])
        rows = rows[cell_of_row[rows] >= 0]
        codes = cell_of_row[rows]
        order = np.argsort(codes, kind="stable")
        rows, codes = rows[order], codes[order]
        bounds = np.searchsorted(codes, np.arange(len(cells) + 1))
        by_group_b: dict[str, list[float]] = {}
        by_group_c: dict[str, list[float]] = {}
        for index in range(len(cells)):
            block = rows[bounds[index] : bounds[index + 1]]
            if block.size == 0:
                continue
            block_labels = labels[block]
            by_group_b.setdefault(groups[index], []).append(
                average_precision(block_labels, baseline[block])
            )
            by_group_c.setdefault(groups[index], []).append(
                average_precision(block_labels, candidate[block])
            )
        deltas[draw] = _macro(by_group_c) - _macro(by_group_b)

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "baseline": point_baseline,
        "candidate": point_candidate,
        "delta": point_candidate - point_baseline,
        "delta_se": float(deltas.std(ddof=1)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_bootstrap": n_bootstrap,
    }


def _self_check(frame: pd.DataFrame, cells: list[Cell]) -> None:
    """Fail fast if the fast AUPRC diverges from scikit-learn on real data."""
    labels = frame["label"].to_numpy().astype(np.int8)
    scores = frame["score_candidate"].to_numpy()
    for cell in cells[:3]:
        mine = average_precision(labels[cell.rows], scores[cell.rows])
        theirs = average_precision_score(labels[cell.rows], scores[cell.rows])
        if not np.isclose(mine, theirs, atol=SELF_CHECK_TOLERANCE):
            raise AssertionError(
                f"AUPRC mismatch on {cell.name}: {mine} vs sklearn {theirs}"
            )


def _load(path: str, column: str, suffix: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if column == "minus_llr_avg" and column not in frame:
        frame[column] = -(frame["llr_fwd"] + frame["llr_rc"]) / 2
    keep = VARIANT_KEY + ["label", "subset"]
    keep += [c for c in ("match_group", "mavedb_urn") if c in frame]
    return frame[keep + [column]].rename(columns={column: f"score_{suffix}"})


def compare(
    dataset: str,
    baseline_path: str,
    candidate_path: str,
    column: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    baseline = _load(baseline_path, column, "baseline")
    candidate = _load(candidate_path, column, "candidate")
    merged = baseline.merge(
        candidate[VARIANT_KEY + ["score_candidate"]], on=VARIANT_KEY, validate="1:1"
    ).reset_index(drop=True)
    if len(merged) != len(baseline):
        raise ValueError("bundles do not cover identical variants")
    if dataset == "mendelian_traits":
        cells = _mendelian_cells(merged)
        unit = merged["match_group"].to_numpy()
    elif dataset == "sge":
        cells = _sge_cells(merged)
        unit = np.arange(len(merged))
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    _self_check(merged, cells)
    return paired_delta(
        merged, cells, unit=unit, n_bootstrap=n_bootstrap, seed=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mendelian_traits", "sge"), required=True)
    parser.add_argument("--baseline-bundle", required=True)
    parser.add_argument("--candidate-bundle", required=True)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {
        "dataset": args.dataset,
        "baseline_id": args.baseline_id,
        "candidate_id": args.candidate_id,
        "readouts": {
            "zero_shot_minus_llr_avg": compare(
                args.dataset,
                f"{args.baseline_bundle.rstrip('/')}/scores.parquet",
                f"{args.candidate_bundle.rstrip('/')}/scores.parquet",
                "minus_llr_avg",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            ),
            "linear_probe": compare(
                args.dataset,
                f"{args.baseline_bundle.rstrip('/')}/probe_predictions.parquet",
                f"{args.candidate_bundle.rstrip('/')}/probe_predictions.parquet",
                "probe_score",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            ),
        },
    }
    payload = json.dumps(results, indent=2)
    if args.output.startswith("s3://"):
        import fsspec

        with fsspec.open(args.output, "w") as handle:
            handle.write(payload)
    else:
        with open(args.output, "w") as handle:
            handle.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
