"""Offline integrity and semantic validation for the released m5.1 snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

EXPECTED_ARCHITECTURE = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "vocab_size": 7,
    "max_position_embeddings": 256,
    "hidden_size": 1920,
    "intermediate_size": 7680,
    "num_hidden_layers": 19,
    "num_attention_heads": 15,
    "num_key_value_heads": 15,
    "head_dim": 128,
    "bos_token_id": 2,
    "pad_token_id": 0,
    "eos_token_id": None,
}
EXPECTED_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "a": 3,
    "c": 4,
    "g": 5,
    "t": 6,
}


@dataclass(frozen=True)
class ModelAssetReport:
    """Validated identity of one local m5.1 snapshot."""

    repo_id: str
    revision: str
    files: dict[str, str]
    parameters: int
    sequence_tokens: int
    supervised_targets: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_assets(
    assets_path: str | Path, snapshot_path: str | Path
) -> ModelAssetReport:
    """Verify byte identity and the exact architecture/tokenizer contract."""
    with Path(assets_path).open("rb") as handle:
        model = tomllib.load(handle).get("model")
    if model is None:
        raise ValueError("assets manifest is missing [model]")
    if not isinstance(model, dict):
        raise TypeError("assets manifest [model] must be a table")
    repo_id = model.get("repo_id")
    revision = model.get("revision")
    expected_files = model.get("files")
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise ValueError("model.repo_id must be a Hugging Face repository id")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model.revision must be a 40-character commit SHA")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("assets manifest is missing [model.files]")

    snapshot = Path(snapshot_path)
    observed_files: dict[str, str] = {}
    for name, expected_hash in expected_files.items():
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"model snapshot is missing {name!r}")
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"model asset {name!r} SHA-256 mismatch: "
                f"expected {expected_hash}, got {observed_hash}"
            )
        observed_files[name] = observed_hash

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    for field, expected in EXPECTED_ARCHITECTURE.items():
        if config.get(field) != expected:
            raise ValueError(
                f"config.json field {field!r}: expected {expected!r}, "
                f"got {config.get(field)!r}"
            )

    tokenizer_json = json.loads(
        (snapshot / "tokenizer.json").read_text(encoding="utf-8")
    )
    if tokenizer_json.get("model", {}).get("vocab") != EXPECTED_VOCAB:
        raise ValueError("tokenizer vocabulary does not match the released m5.1 map")
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    upper = tokenizer.encode("ACGT").ids
    lower = tokenizer.encode("acgt").ids
    if upper != [2, 3, 4, 5, 6] or lower != upper:
        raise ValueError(
            "tokenizer must prepend BOS and case-normalize A/C/G/T to ids 3/4/5/6"
        )
    if tokenizer.encode("NnRr").ids != [2, 1, 1, 1, 1]:
        raise ValueError("tokenizer must map ambiguous DNA bases to [UNK]")
    sequence_tokens = len(tokenizer.encode("A" * 255).ids)
    if sequence_tokens != 256:
        raise ValueError(
            f"tokenizer must produce BOS + 255 bases, got {sequence_tokens} tokens"
        )

    return ModelAssetReport(
        repo_id=repo_id,
        revision=revision,
        files=observed_files,
        parameters=1_120_772_224,
        sequence_tokens=sequence_tokens,
        supervised_targets=255,
    )
