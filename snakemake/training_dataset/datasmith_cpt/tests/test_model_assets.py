from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from marin_dna_datasmith_cpt.model_assets import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_VOCAB,
    validate_model_assets,
)
from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordLevel


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_snapshot(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps(EXPECTED_ARCHITECTURE))
    (root / "model.safetensors").write_bytes(b"fixture weights")
    (root / "special_tokens_map.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")

    tokenizer = Tokenizer(WordLevel(vocab=EXPECTED_VOCAB, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(".{1}"), behavior="isolated")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        pair="$A $B",
        special_tokens=[("[BOS]", 2)],
    )
    tokenizer.save(str(root / "tokenizer.json"))
    return root


def _write_manifest(path: Path, snapshot: Path) -> Path:
    files = {
        name: _sha256(snapshot / name)
        for name in (
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    lines = [
        "[model]",
        'repo_id = "example/model"',
        f'revision = "{"a" * 40}"',
        "",
        "[model.files]",
        *[f'"{name}" = "{digest}"' for name, digest in files.items()],
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_model_snapshot_contract(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    manifest = _write_manifest(tmp_path / "assets.toml", snapshot)

    report = validate_model_assets(manifest, snapshot)

    assert report.parameters == 1_120_772_224
    assert report.sequence_tokens == 256
    assert report.supervised_targets == 255


def test_model_snapshot_rejects_byte_drift(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "snapshot")
    manifest = _write_manifest(tmp_path / "assets.toml", snapshot)
    (snapshot / "config.json").write_text("{}")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_model_assets(manifest, snapshot)
