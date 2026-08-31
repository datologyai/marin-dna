"""Immutable public-asset and training-sequence contracts."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_BASES = frozenset("ACGTacgt")
SEQUENCE_LENGTH = 255


@dataclass(frozen=True)
class StreamSpec:
    """One pinned public training stream."""

    name: str
    repo_id: str
    revision: str
    sequence_field: str
    expected_rows: int
    expected_shards: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stream name must not be empty")
        if self.repo_id.count("/") != 1:
            raise ValueError(f"invalid Hugging Face dataset id: {self.repo_id!r}")
        if not REVISION_RE.fullmatch(self.revision):
            raise ValueError(
                f"stream {self.name!r} revision must be a 40-character commit SHA"
            )
        if self.sequence_field not in {"seq", "sequence"}:
            raise ValueError(
                f"stream {self.name!r} has unsupported sequence field "
                f"{self.sequence_field!r}"
            )
        if self.expected_rows < 1 or self.expected_shards < 1:
            raise ValueError(
                f"stream {self.name!r} row and shard counts must be positive"
            )


def load_stream_specs(path: str | Path) -> dict[str, StreamSpec]:
    """Load and validate every training stream from ``assets.toml``."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, dict) or not raw_streams:
        raise ValueError("assets manifest must contain a non-empty [streams] table")

    specs = {
        name: StreamSpec(name=name, **values) for name, values in raw_streams.items()
    }
    if len(specs) != len(raw_streams):
        raise ValueError("assets manifest contains duplicate stream names")
    return specs


def validate_sequence(value: object, *, source: str) -> str:
    """Return a valid 255-base mixed-case DNA sequence or fail loudly."""
    if not isinstance(value, str):
        raise TypeError(f"{source}: sequence must be a string")
    if len(value) != SEQUENCE_LENGTH:
        raise ValueError(
            f"{source}: sequence length must be {SEQUENCE_LENGTH}, got {len(value)}"
        )
    invalid = sorted(set(value).difference(CANONICAL_BASES))
    if invalid:
        raise ValueError(f"{source}: sequence contains invalid bases: {invalid}")
    return value
