from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from marin_dna_evals import cell


def _mendelian_fixture() -> pd.DataFrame:
    rows = []
    for index in range(1, 31):
        chrom = str((index - 1) % 3 + 1)
        rows.extend(
            [
                {
                    "chrom": chrom,
                    "pos": index * 10,
                    "ref": "A",
                    "alt": "C",
                    "label": True,
                    "subset": "missense_variant",
                    "match_group": f"g{index}",
                },
                {
                    "chrom": chrom,
                    "pos": index * 10 + 1,
                    "ref": "G",
                    "alt": "T",
                    "label": False,
                    "subset": "missense_variant",
                    "match_group": f"g{index}",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_dataset_specs_match_the_released_m51_config() -> None:
    assert cell.DATASET_SPECS["mendelian_traits"].revision == (
        "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
    )
    assert cell.DATASET_SPECS["sge"].revision == (
        "225d3d1ea32a4af547891b13c33b5e92a5aae849"
    )
    assert cell.DATASET_SPECS["mendelian_traits"].supports_probe
    assert not cell.DATASET_SPECS["sge"].supports_probe


def test_sge_probe_is_not_silently_added_to_the_released_protocol() -> None:
    with pytest.raises(ValueError, match="not part of the released m5.1 protocol"):
        cell.CellConfig(
            dataset="sge",
            checkpoint_uri="checkpoint",
            genome_uri="genome",
            run_probe=True,
        )


def test_cell_emits_separate_zero_shot_probe_and_provenance_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _mendelian_fixture()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    monkeypatch.setattr(
        cell,
        "_stage_inputs_isolated",
        lambda _spec, _checkpoint_uri, _dataset_uri, _destination: (
            checkpoint,
            dataset,
        ),
    )

    def fake_scores(**kwargs: object) -> pd.DataFrame:
        assert kwargs["context_size"] == 255
        assert kwargs["batch_size"] == 64
        assert kwargs["num_workers"] == 4
        assert kwargs["torch_compile"] is True
        assert kwargs["bf16"] is True
        assert kwargs["rc"] is True
        assert kwargs["return_embeddings"] is True
        assert kwargs["eval_accumulation_steps"] == 8
        n = len(dataset)
        return pd.DataFrame(
            {
                "llr_fwd": np.linspace(-2, 2, n),
                "llr_rc": np.linspace(-1, 1, n),
                "jsd_fwd": np.linspace(0, 1, n),
                "jsd_rc": np.linspace(0.1, 1.1, n),
                "emb_ref": [np.array([1, 2], dtype=np.float16)] * n,
                "emb_alt": [np.array([2, 3], dtype=np.float16)] * n,
            }
        )

    monkeypatch.setattr(cell, "compute_variant_scores", fake_scores)

    def fake_probe(
        frame: pd.DataFrame, **_kwargs: object
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        predictions = frame.drop(columns=["emb_ref", "emb_alt"]).copy()
        predictions["probe_score"] = np.where(predictions["label"], 0.9, 0.1)
        return predictions, {"missense_variant": {"C": 1.0}}

    monkeypatch.setattr(cell, "run_subset_probes", fake_probe)

    outputs = cell.CellOutputs(
        scores=str(tmp_path / "scores.parquet"),
        zero_shot_metrics=str(tmp_path / "zero_shot.parquet"),
        provenance=str(tmp_path / "provenance.json"),
        probe_predictions=str(tmp_path / "probe_predictions.parquet"),
        probe_metrics=str(tmp_path / "probe_metrics.parquet"),
        probe_classifiers=str(tmp_path / "probe_classifiers.joblib"),
    )
    config = cell.CellConfig(
        dataset="mendelian_traits",
        checkpoint_uri=str(checkpoint),
        genome_uri="s3://example/genome.fa.gz",
        adapter_commit="a" * 40,
        n_bootstrap=0,
        run_probe=True,
        probe_min_variants=1,
        probe_min_chroms=3,
        probe_n_min=1,
    )

    provenance = cell.run_cell(config, outputs)

    scores = pd.read_parquet(outputs.scores)
    zero_shot = pd.read_parquet(outputs.zero_shot_metrics)
    probe_metrics = pd.read_parquet(outputs.probe_metrics)
    stored_provenance = json.loads(Path(outputs.provenance).read_text())
    assert len(scores) == len(dataset)
    assert {"emb_ref", "emb_alt", "llr_fwd", "llr_rc"} <= set(scores.columns)
    assert {"minus_llr_avg", "jsd_avg"} <= set(zero_shot["score_type"])
    assert set(probe_metrics["score_type"]) == {"probe_score", "minus_llr_avg"}
    assert provenance["rows"] == len(dataset)
    assert stored_provenance["dataset"]["revision"] == (
        "4aed58e50c5dea0b878a665007af2ef9e5108e9f"
    )
    assert set(stored_provenance["artifacts"]) == {
        "scores",
        "zero_shot_metrics",
        "probe_predictions",
        "probe_metrics",
        "probe_classifiers",
    }
    assert all(
        len(artifact["sha256"]) == 64
        for artifact in stored_provenance["artifacts"].values()
    )


def test_dataset_mirror_is_checksum_and_row_count_validated(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.parquet"
    dataset = _mendelian_fixture()
    dataset.to_parquet(dataset_path, index=False)
    spec = replace(
        cell.DATASET_SPECS["mendelian_traits"],
        rows=len(dataset),
        sha256=cell._sha256_uri(str(dataset_path)),
    )

    loaded = cell._load_dataset_frame(spec, str(dataset_path))
    assert loaded.equals(dataset)

    bad_spec = replace(spec, sha256="0" * 64)
    with pytest.raises(ValueError, match="mirror SHA-256"):
        cell._load_dataset_frame(bad_spec, str(dataset_path))


def test_remote_input_staging_finishes_in_an_isolated_process(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    dataset_path = tmp_path / "source.parquet"
    dataset = _mendelian_fixture()
    dataset.to_parquet(dataset_path, index=False)
    spec = replace(
        cell.DATASET_SPECS["mendelian_traits"],
        rows=len(dataset),
        sha256=cell._sha256_uri(str(dataset_path)),
    )

    staged_checkpoint, staged_dataset = cell._stage_inputs_isolated(
        spec,
        str(checkpoint),
        str(dataset_path),
        tmp_path / "staged",
    )

    assert staged_checkpoint == checkpoint
    assert staged_dataset.equals(dataset)
