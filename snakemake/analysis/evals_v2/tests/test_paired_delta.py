from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from marin_dna_evals import paired_delta


def _mendelian_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subset in ("splicing", "missense_variant"):
        for group in range(12):
            for member in range(4):
                label = member == 0
                rows.append(
                    {
                        "chrom": "1",
                        "pos": len(rows),
                        "ref": "A",
                        "alt": "G",
                        "label": label,
                        "subset": subset,
                        "match_group": f"{subset}-{group}",
                        "llr_fwd": rng.normal(-1.0 if label else 0.0),
                        "llr_rc": rng.normal(-1.0 if label else 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_average_precision_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        labels = rng.integers(0, 2, 60).astype(np.int8)
        if labels.all() or not labels.any():
            continue
        scores = rng.normal(size=60)
        assert paired_delta.average_precision(labels, scores) == pytest.approx(
            average_precision_score(labels, scores)
        )


def test_paired_delta_is_zero_for_identical_scores(tmp_path) -> None:
    frame = _mendelian_frame()
    frame["minus_llr_avg"] = -(frame["llr_fwd"] + frame["llr_rc"]) / 2
    path = tmp_path / "scores.parquet"
    frame.to_parquet(path)

    result = paired_delta.compare(
        "mendelian_traits", str(path), str(path), "minus_llr_avg", n_bootstrap=25, seed=0
    )
    assert result["delta"] == pytest.approx(0.0)
    assert result["delta_se"] == pytest.approx(0.0)
    assert result["baseline"] == pytest.approx(result["candidate"])


def test_paired_delta_detects_a_better_candidate(tmp_path) -> None:
    frame = _mendelian_frame()
    frame["minus_llr_avg"] = -(frame["llr_fwd"] + frame["llr_rc"]) / 2
    baseline = tmp_path / "baseline.parquet"
    frame.to_parquet(baseline)

    better = frame.copy()
    better["minus_llr_avg"] = np.where(
        better["label"], better["minus_llr_avg"] + 5.0, better["minus_llr_avg"]
    )
    candidate = tmp_path / "candidate.parquet"
    better.to_parquet(candidate)

    result = paired_delta.compare(
        "mendelian_traits",
        str(baseline),
        str(candidate),
        "minus_llr_avg",
        n_bootstrap=50,
        seed=0,
    )
    assert result["candidate"] == pytest.approx(1.0)
    assert result["delta"] > 0.3
    assert result["ci95_low"] > 0.0


def test_small_match_group_subsets_are_dropped() -> None:
    frame = _mendelian_frame()
    tiny = frame[frame.subset == "splicing"].head(8).copy()
    tiny["subset"] = "mature_miRNA_variant"
    combined = pd.concat([frame, tiny], ignore_index=True)
    cells = paired_delta._mendelian_cells(combined)
    assert {cell.name for cell in cells} == {"splicing", "missense_variant"}
