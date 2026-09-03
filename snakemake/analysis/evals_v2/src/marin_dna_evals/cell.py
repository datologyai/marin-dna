"""Run pinned MarinDNA scoring with the DataSmith fixed-C linear probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import fsspec
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

from marin_dna_evals.conservation import (
    REQUIRED_VARIANT_COLUMNS,
    SGE_VARIANT_COLUMNS,
)
from marin_dna_evals.grouped_vep_metrics import compute_grouped_vep_metrics
from marin_dna_evals.inference import compute_variant_scores
from marin_dna_evals.metrics import (
    compute_sge_metrics,
    compute_sge_probe_metrics,
    per_chrom_ap_table,
)
from marin_dna_evals.variant_probe import (
    DEFAULT_FIXED_PROBE_C,
    run_subset_probes_fixed_c,
)

DatasetName = Literal["mendelian_traits", "sge"]
EvalProtocol = Literal["matched_pair", "sge"]


@dataclass(frozen=True)
class DatasetSpec:
    """One immutable public evaluation cohort and its scoring protocol."""

    name: DatasetName
    repo_id: str
    revision: str
    split: str
    protocol: EvalProtocol
    required_columns: tuple[str, ...]
    supports_probe: bool
    rows: int
    sha256: str


DATASET_SPECS: dict[DatasetName, DatasetSpec] = {
    "mendelian_traits": DatasetSpec(
        name="mendelian_traits",
        repo_id="bolinas-dna/evals_mendelian_traits",
        revision="4aed58e50c5dea0b878a665007af2ef9e5108e9f",
        split="train",
        protocol="matched_pair",
        required_columns=REQUIRED_VARIANT_COLUMNS,
        supports_probe=True,
        rows=16_140,
        sha256="919abf532debd9f5e339b94cd4eef9868cbfa92dbe4fc57a0b1398da481f469f",
    ),
    "sge": DatasetSpec(
        name="sge",
        repo_id="bolinas-dna/evals_sge",
        revision="225d3d1ea32a4af547891b13c33b5e92a5aae849",
        split="train",
        protocol="sge",
        required_columns=SGE_VARIANT_COLUMNS,
        supports_probe=True,
        rows=23_853,
        sha256="781a22c4d8a1eddc93ff95d1d45f8b5aa56adf38e9e96ff4cc944968de5d0450",
    ),
}

EMBEDDING_COLUMNS: tuple[str, str] = ("emb_ref", "emb_alt")


@dataclass(frozen=True)
class CellConfig:
    """Output-affecting protocol plus execution-only inference settings."""

    dataset: DatasetName
    checkpoint_uri: str
    genome_uri: str
    dataset_uri: str | None = None
    model_id: str = "marin-dna/marin-dna-exp135-m5.1"
    adapter_commit: str = "unknown"
    window_size: int = 255
    batch_size: int = 64
    num_workers: int = 4
    eval_accumulation_steps: int = 8
    n_bootstrap: int = 1000
    bootstrap_seed: int = 0
    run_probe: bool = False
    retain_embeddings: bool = False
    probe_min_variants: int = 300
    probe_min_chroms: int = 3
    probe_n_min: int = 30
    probe_n_jobs: int = 4
    probe_fixed_c: float = DEFAULT_FIXED_PROBE_C

    def __post_init__(self) -> None:
        spec = DATASET_SPECS[self.dataset]
        if self.window_size != 255:
            raise ValueError("released m5.1 evaluation requires a 255-base window")
        if self.run_probe and not spec.supports_probe:
            raise ValueError(
                f"{self.dataset} probe is not part of the released m5.1 protocol"
            )
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.eval_accumulation_steps < 1:
            raise ValueError("eval_accumulation_steps must be positive")
        if self.n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")
        if self.probe_n_jobs == 0:
            raise ValueError("probe_n_jobs must not be zero")
        if not np.isfinite(self.probe_fixed_c) or self.probe_fixed_c <= 0:
            raise ValueError("probe_fixed_c must be finite and positive")


@dataclass(frozen=True)
class CellOutputs:
    """Tracked outputs for one model-by-dataset evaluation cell."""

    scores: str
    zero_shot_metrics: str
    provenance: str
    probe_predictions: str | None = None
    probe_metrics: str | None = None
    probe_classifiers: str | None = None

    def __post_init__(self) -> None:
        probe_outputs = (
            self.probe_predictions,
            self.probe_metrics,
            self.probe_classifiers,
        )
        if any(value is not None for value in probe_outputs) and not all(
            value is not None for value in probe_outputs
        ):
            raise ValueError("probe outputs must be provided together")


def _sha256_uri(uri: str) -> str:
    digest = hashlib.sha256()
    with fsspec.open(uri, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset_frame(spec: DatasetSpec, dataset_uri: str | None) -> pd.DataFrame:
    if dataset_uri is None:
        frame = load_dataset(
            spec.repo_id,
            revision=spec.revision,
            split=spec.split,
        ).to_pandas()
    else:
        actual_sha256 = _sha256_uri(dataset_uri)
        if actual_sha256 != spec.sha256:
            raise ValueError(
                f"{spec.name} mirror SHA-256 {actual_sha256} != {spec.sha256}"
            )
        frame = pd.read_parquet(dataset_uri)
    if len(frame) != spec.rows:
        raise ValueError(f"{spec.name} row count {len(frame)} != {spec.rows}")
    missing = set(spec.required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.name} missing required columns: {sorted(missing)}")
    return frame


def _stage_checkpoint(uri: str, destination: Path) -> Path:
    source = Path(uri)
    if source.is_dir():
        return source

    fs, root = fsspec.core.url_to_fs(uri)
    files = [path for path in fs.find(root) if not fs.isdir(path)]
    if not files:
        raise FileNotFoundError(f"checkpoint prefix is empty: {uri}")
    for remote_path in files:
        relative = Path(remote_path).relative_to(root)
        local_path = destination / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, str(local_path))
    return destination


def _stage_inputs(
    spec: DatasetSpec,
    checkpoint_uri: str,
    dataset_uri: str | None,
    destination: str,
) -> tuple[str, str]:
    """Materialize and validate remote inputs inside one short-lived process."""
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _stage_checkpoint(checkpoint_uri, destination_path / "checkpoint")
    dataset = _load_dataset_frame(spec, dataset_uri)
    dataset_path = destination_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    return str(checkpoint_path), str(dataset_path)


def _stage_inputs_isolated(
    spec: DatasetSpec,
    checkpoint_uri: str,
    dataset_uri: str | None,
    destination: Path,
) -> tuple[Path, pd.DataFrame]:
    """Stage S3 inputs without leaking fsspec state into DataLoader forks.

    fsspec owns an asyncio thread. If this process opens S3 before PyTorch
    forks data-loader workers, the children inherit a stale event loop and can
    wait forever. A spawned staging process restores the process boundary used
    by the upstream pipeline while leaving the scoring protocol unchanged.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        checkpoint_string, dataset_string = executor.submit(
            _stage_inputs,
            spec,
            checkpoint_uri,
            dataset_uri,
            str(destination),
        ).result()
    return Path(checkpoint_string), pd.read_parquet(dataset_string)


def _derived_zero_shot_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    for column in ("llr_fwd", "llr_rc", "jsd_fwd", "jsd_rc"):
        if column not in scores.columns:
            raise ValueError(f"score bundle missing {column!r}")
    derived = pd.DataFrame(index=scores.index)
    derived["minus_llr_fwd"] = -scores["llr_fwd"]
    derived["jsd_fwd"] = scores["jsd_fwd"]
    derived["minus_llr_rc"] = -scores["llr_rc"]
    derived["jsd_rc"] = scores["jsd_rc"]
    derived["minus_llr_avg"] = -(scores["llr_fwd"] + scores["llr_rc"]) / 2
    derived["jsd_avg"] = (scores["jsd_fwd"] + scores["jsd_rc"]) / 2
    return derived, list(derived.columns)


def _zero_shot_metrics(
    spec: DatasetSpec,
    score_bundle: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    derived, score_columns = _derived_zero_shot_scores(score_bundle)
    if spec.protocol == "matched_pair":
        metrics = compute_grouped_vep_metrics(
            dataset=score_bundle[list(REQUIRED_VARIANT_COLUMNS)],
            scores=derived,
            score_columns=score_columns,
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
    else:
        metrics = compute_sge_metrics(
            dataset=score_bundle[list(SGE_VARIANT_COLUMNS)],
            scores=derived,
            score_columns=score_columns,
            n_bootstrap=n_bootstrap,
            rng=seed,
        )
    return metrics


def _run_probe(
    score_bundle: pd.DataFrame,
    config: CellConfig,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    predictions, classifiers = run_subset_probes_fixed_c(
        score_bundle,
        feature_combo="concat_ref_delta",
        fixed_c=config.probe_fixed_c,
        min_variants=config.probe_min_variants,
        min_chroms=config.probe_min_chroms,
        n_jobs=config.probe_n_jobs,
    )
    if DATASET_SPECS[config.dataset].protocol == "sge":
        metrics = compute_sge_probe_metrics(
            predictions,
            "minus_llr",
            n_bootstrap=config.n_bootstrap,
            rng=config.bootstrap_seed,
        )
    else:
        predictions["minus_llr_avg"] = (
            -(predictions["llr_fwd"] + predictions["llr_rc"]) / 2
        )
        metrics = per_chrom_ap_table(
            predictions,
            ["probe_score", "minus_llr_avg"],
            n_bootstrap=config.n_bootstrap,
            rng=config.bootstrap_seed,
            n_min=config.probe_n_min,
        )
    return predictions, classifiers, metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish(local_path: Path, uri: str) -> None:
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    fs.put(str(local_path), path)


def run_cell(config: CellConfig, outputs: CellOutputs) -> dict[str, object]:
    """Run scoring, metrics, and the optional fixed-C probe."""
    spec = DATASET_SPECS[config.dataset]
    if config.run_probe and outputs.probe_predictions is None:
        raise ValueError("run_probe=True requires all probe output paths")
    if not config.run_probe and outputs.probe_predictions is not None:
        raise ValueError("probe output paths require run_probe=True")

    with tempfile.TemporaryDirectory(prefix="marin-dna-eval-") as temp_dir_string:
        temp_dir = Path(temp_dir_string)
        checkpoint_path, dataset = _stage_inputs_isolated(
            spec,
            config.checkpoint_uri,
            config.dataset_uri,
            temp_dir,
        )
        scores = compute_variant_scores(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            genome_path=config.genome_uri,
            context_size=config.window_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_transform_on_the_fly=True,
            torch_compile=True,
            bf16=True,
            rc=True,
            return_embeddings=config.run_probe or config.retain_embeddings,
            eval_accumulation_steps=config.eval_accumulation_steps,
        )
        if len(scores) != len(dataset):
            raise ValueError(
                f"score row count {len(scores)} != dataset row count {len(dataset)}"
            )
        score_bundle = pd.concat(
            [dataset.reset_index(drop=True), scores.reset_index(drop=True)], axis=1
        )
        zero_shot_metrics = _zero_shot_metrics(
            spec,
            score_bundle,
            n_bootstrap=config.n_bootstrap,
            seed=config.bootstrap_seed,
        )
        for frame in (zero_shot_metrics,):
            frame["model"] = config.model_id
            frame["dataset"] = spec.name
            frame["split"] = spec.split

        local_artifacts: dict[str, Path] = {}
        output_uris: dict[str, str] = {}
        if config.run_probe:
            assert outputs.probe_predictions is not None
            assert outputs.probe_metrics is not None
            assert outputs.probe_classifiers is not None
            predictions, classifiers, probe_metrics = _run_probe(score_bundle, config)
            probe_metrics["model"] = config.model_id
            probe_metrics["dataset"] = spec.name
            probe_metrics["split"] = spec.split
            local_predictions = temp_dir / "probe_predictions.parquet"
            local_probe_metrics = temp_dir / "probe_metrics.parquet"
            local_classifiers = temp_dir / "probe_classifiers.joblib"
            predictions.to_parquet(local_predictions, index=False)
            probe_metrics.to_parquet(local_probe_metrics, index=False)
            joblib.dump(classifiers, local_classifiers)
            local_artifacts.update(
                {
                    "probe_predictions": local_predictions,
                    "probe_metrics": local_probe_metrics,
                    "probe_classifiers": local_classifiers,
                }
            )
            output_uris.update(
                {
                    "probe_predictions": outputs.probe_predictions,
                    "probe_metrics": outputs.probe_metrics,
                    "probe_classifiers": outputs.probe_classifiers,
                }
            )

        archived_scores = (
            score_bundle
            if config.retain_embeddings
            else score_bundle.drop(columns=list(EMBEDDING_COLUMNS), errors="ignore")
        )
        local_scores = temp_dir / "scores.parquet"
        local_zero_shot = temp_dir / "zero_shot_metrics.parquet"
        archived_scores.to_parquet(local_scores, index=False)
        zero_shot_metrics.to_parquet(local_zero_shot, index=False)
        local_artifacts.update(
            {
                "scores": local_scores,
                "zero_shot_metrics": local_zero_shot,
            }
        )
        output_uris.update(
            {
                "scores": outputs.scores,
                "zero_shot_metrics": outputs.zero_shot_metrics,
            }
        )

        provenance: dict[str, object] = {
            "format_version": 1,
            "model_id": config.model_id,
            "checkpoint_uri": config.checkpoint_uri,
            "genome_uri": config.genome_uri,
            "adapter_commit": config.adapter_commit,
            "dataset": asdict(spec),
            "config": asdict(config),
            "rows": len(score_bundle),
            "score_columns": list(archived_scores.columns),
            "artifacts": {
                name: {
                    "uri": output_uris[name],
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in local_artifacts.items()
            },
        }
        local_provenance = temp_dir / "provenance.json"
        local_provenance.write_text(json.dumps(provenance, indent=2) + "\n")

        for name, path in local_artifacts.items():
            _publish(path, output_uris[name])
        _publish(local_provenance, outputs.provenance)
        return provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--genome-uri", required=True)
    parser.add_argument("--dataset-uri")
    parser.add_argument("--model-id", default="marin-dna/marin-dna-exp135-m5.1")
    parser.add_argument("--adapter-commit", required=True)
    parser.add_argument("--scores-output", required=True)
    parser.add_argument("--zero-shot-metrics-output", required=True)
    parser.add_argument("--provenance-output", required=True)
    parser.add_argument("--run-probe", action="store_true")
    parser.add_argument("--retain-embeddings", action="store_true")
    parser.add_argument("--probe-predictions-output")
    parser.add_argument("--probe-metrics-output")
    parser.add_argument("--probe-classifiers-output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-n-jobs", type=int, default=4)
    parser.add_argument("--probe-fixed-c", type=float, default=DEFAULT_FIXED_PROBE_C)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = CellConfig(
        dataset=cast(DatasetName, args.dataset),
        checkpoint_uri=args.checkpoint_uri,
        genome_uri=args.genome_uri,
        dataset_uri=args.dataset_uri,
        model_id=args.model_id,
        adapter_commit=args.adapter_commit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        probe_n_jobs=args.probe_n_jobs,
        probe_fixed_c=args.probe_fixed_c,
        n_bootstrap=args.n_bootstrap,
        run_probe=args.run_probe,
        retain_embeddings=args.retain_embeddings,
    )
    outputs = CellOutputs(
        scores=args.scores_output,
        zero_shot_metrics=args.zero_shot_metrics_output,
        provenance=args.provenance_output,
        probe_predictions=args.probe_predictions_output,
        probe_metrics=args.probe_metrics_output,
        probe_classifiers=args.probe_classifiers_output,
    )
    provenance = run_cell(config, outputs)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
