"""Stream pinned JSONL-Zstandard shards into Universe-compatible LitData."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

import fsspec
import orjson
import zstandard

from marin_dna_datasmith_cpt.contracts import (
    CANONICAL_BASES,
    StreamSpec,
    validate_sequence,
)

SHARD_RE = re.compile(r"^data/train/shard_(\d{4})\.jsonl\.zst$")
HF_TREE_URL = "https://huggingface.co/api/datasets/{repo_id}/tree/{revision}"
HF_RESOLVE_URL = (
    "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}?download=true"
)


@dataclass(frozen=True)
class Shard:
    """One immutable compressed source shard."""

    path: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ShardReport:
    """Validation and conversion statistics for one source shard."""

    stream: str
    path: str
    source_sha256: str
    source_bytes: int
    rows: int
    bases: int
    uppercase_bases: int
    lowercase_bases: int
    unknown_bases: int


class _HashingReader(io.RawIOBase):
    """Update a SHA-256 digest as compressed bytes are consumed."""

    def __init__(self, raw: BinaryIO) -> None:
        self._raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        if data:
            self.digest.update(data)
            self.bytes_read += len(data)
        return data

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def discover_shards(spec: StreamSpec) -> list[Shard]:
    """Resolve and validate the complete pinned HF shard inventory."""
    query = urllib.parse.urlencode(
        {"recursive": "true", "expand": "false", "limit": "1000"}
    )
    url = HF_TREE_URL.format(repo_id=spec.repo_id, revision=spec.revision)
    with urllib.request.urlopen(f"{url}?{query}") as response:
        entries = json.load(response)

    shards: list[Shard] = []
    for entry in entries:
        path = entry.get("path", "")
        match = SHARD_RE.fullmatch(path)
        if not match:
            continue
        lfs = entry.get("lfs") or {}
        sha256 = lfs.get("oid")
        size = entry.get("size")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"{spec.name}: shard {path!r} has no valid LFS SHA-256")
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"{spec.name}: shard {path!r} has no valid size")
        download_url = HF_RESOLVE_URL.format(
            repo_id=spec.repo_id,
            revision=spec.revision,
            path=urllib.parse.quote(path),
        )
        shards.append(Shard(path=path, url=download_url, size=size, sha256=sha256))

    shards.sort(key=lambda shard: shard.path)
    expected_paths = [
        f"data/train/shard_{index:04d}.jsonl.zst"
        for index in range(spec.expected_shards)
    ]
    observed_paths = [shard.path for shard in shards]
    if observed_paths != expected_paths:
        raise ValueError(
            f"{spec.name}: expected the contiguous {spec.expected_shards}-shard "
            f"inventory, got {len(shards)} shards"
        )
    return shards


def _join_uri(root: str, name: str) -> str:
    return f"{root.rstrip('/')}/{name.lstrip('/')}"


def _write_json(uri: str, value: object) -> None:
    with fsspec.open(uri, "wb") as handle:
        handle.write(orjson.dumps(value, option=orjson.OPT_INDENT_2))
        handle.write(b"\n")


def convert_shard(
    shard: Shard,
    *,
    stream: str,
    sequence_field: str,
    report_root: str,
) -> Iterator[dict[str, dict[str, str]]]:
    """Validate one compressed shard and yield LitData text records."""
    rows = 0
    uppercase_bases = 0
    lowercase_bases = 0
    unknown_bases = 0
    report_name = f"{Path(shard.path).name}.report.json"

    with fsspec.open(shard.url, "rb") as source:
        hashing_source = _HashingReader(source)
        with (
            zstandard.ZstdDecompressor().stream_reader(
                hashing_source, closefd=False
            ) as decompressed,
            io.TextIOWrapper(decompressed, encoding="utf-8") as text,
        ):
            for line_number, line in enumerate(text, start=1):
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError as error:
                    raise ValueError(
                        f"{stream}/{shard.path}:{line_number}: invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise TypeError(
                        f"{stream}/{shard.path}:{line_number}: row must be an object"
                    )
                sequence = validate_sequence(
                    row.get(sequence_field),
                    source=f"{stream}/{shard.path}:{line_number}",
                )
                uppercase_bases += sum(base.isupper() for base in sequence)
                lowercase_bases += sum(base.islower() for base in sequence)
                unknown_bases += sum(base not in CANONICAL_BASES for base in sequence)
                rows += 1
                yield {"text": {"content": sequence}}

        while hashing_source.read(1024 * 1024):
            pass
        observed_sha256 = hashing_source.digest.hexdigest()
        source_bytes = hashing_source.bytes_read

    if observed_sha256 != shard.sha256:
        raise ValueError(
            f"{stream}/{shard.path}: compressed SHA-256 mismatch: "
            f"expected {shard.sha256}, got {observed_sha256}"
        )
    if source_bytes != shard.size:
        raise ValueError(
            f"{stream}/{shard.path}: compressed size mismatch: "
            f"expected {shard.size}, got {source_bytes}"
        )

    report = ShardReport(
        stream=stream,
        path=shard.path,
        source_sha256=observed_sha256,
        source_bytes=source_bytes,
        rows=rows,
        bases=rows * 255,
        uppercase_bases=uppercase_bases,
        lowercase_bases=lowercase_bases,
        unknown_bases=unknown_bases,
    )
    _write_json(_join_uri(report_root, report_name), asdict(report))


def convert_stream(
    spec: StreamSpec,
    *,
    output_uri: str,
    report_uri: str,
    num_workers: int,
    chunk_bytes: str,
    compression: str | None,
) -> None:
    """Convert a complete pinned stream into raw-string LitData."""
    from litdata import optimize

    shards = discover_shards(spec)
    optimize(
        partial(
            convert_shard,
            stream=spec.name,
            sequence_field=spec.sequence_field,
            report_root=report_uri,
        ),
        shards,
        output_dir=output_uri,
        chunk_bytes=chunk_bytes,
        compression=compression,
        num_workers=num_workers,
        reorder_files=False,
        mode="overwrite",
        # One input shard expands to a generator of records. LitData 0.2.36
        # cannot checkpoint generator outputs, so retries restart the stream
        # at its immutable output prefix instead of resuming mid-shard.
        use_checkpoint=False,
    )


def finalize_reports(spec: StreamSpec, *, report_uri: str) -> dict[str, object]:
    """Aggregate shard reports and enforce the pinned stream contract."""
    filesystem, root = fsspec.core.url_to_fs(report_uri)
    paths = sorted(filesystem.glob(f"{root.rstrip('/')}/*.report.json"))
    if len(paths) != spec.expected_shards:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_shards} reports, got {len(paths)}"
        )

    reports: list[dict[str, object]] = []
    for path in paths:
        with filesystem.open(path, "rb") as handle:
            reports.append(orjson.loads(handle.read()))

    rows = sum(int(report["rows"]) for report in reports)
    if rows != spec.expected_rows:
        raise ValueError(f"{spec.name}: expected {spec.expected_rows} rows, got {rows}")
    manifest: dict[str, object] = {
        "format_version": 2,
        "stream": asdict(spec),
        "rows": rows,
        "bases": sum(int(report["bases"]) for report in reports),
        "uppercase_bases": sum(int(report["uppercase_bases"]) for report in reports),
        "lowercase_bases": sum(int(report["lowercase_bases"]) for report in reports),
        "unknown_bases": sum(int(report["unknown_bases"]) for report in reports),
        "source_bytes": sum(int(report["source_bytes"]) for report in reports),
        "shards": reports,
    }
    _write_json(_join_uri(report_uri, "manifest.json"), manifest)
    return manifest
