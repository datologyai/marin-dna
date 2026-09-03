"""Stream pinned JSONL-Zstandard shards into Universe-compatible LitData."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

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
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_BLOCK_BYTES = 8 * 1024 * 1024


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
    filtered_rows: int = 0


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


def _download_verified_shard(shard: Shard, destination: Path) -> tuple[str, int]:
    """Download one whole shard before yielding records, with bounded retries."""
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        digest = hashlib.sha256()
        source_bytes = 0
        try:
            # A streaming HTTP file uses one request instead of many range reads.
            with (
                fsspec.open(shard.url, "rb", block_size=0) as source,
                destination.open("wb") as local,
            ):
                while block := source.read(DOWNLOAD_BLOCK_BYTES):
                    local.write(block)
                    digest.update(block)
                    source_bytes += len(block)
        # fsspec backends surface transport failures through several unrelated
        # exception types; this is the single bounded retry boundary.
        except Exception as error:  # noqa: BLE001
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                break
            jitter = int(shard.sha256[:2], 16) % 5
            delay = min(60, 5 * (2 ** (attempt - 1))) + jitter
            print(
                f"{shard.path}: download attempt {attempt}/{DOWNLOAD_ATTEMPTS} "
                f"failed ({error}); retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)
            continue

        observed_sha256 = digest.hexdigest()
        if observed_sha256 != shard.sha256:
            raise ValueError(
                f"compressed SHA-256 mismatch: expected {shard.sha256}, "
                f"got {observed_sha256}"
            )
        if source_bytes != shard.size:
            raise ValueError(
                f"compressed size mismatch: expected {shard.size}, got {source_bytes}"
            )
        return observed_sha256, source_bytes

    raise RuntimeError(
        f"{shard.path}: download failed after {DOWNLOAD_ATTEMPTS} attempts"
    ) from last_error


def convert_shard(
    shard: Shard,
    *,
    spec: StreamSpec,
    report_root: str,
) -> Iterator[dict[str, dict[str, str]]]:
    """Validate one compressed shard and yield LitData text records."""
    stream = spec.name
    sequence_field = spec.sequence_field
    rows = 0
    filtered_rows = 0
    uppercase_bases = 0
    lowercase_bases = 0
    unknown_bases = 0
    report_name = f"{Path(shard.path).name}.report.json"

    with tempfile.TemporaryDirectory(prefix="marin-dna-shard-") as temp_dir:
        local_path = Path(temp_dir) / Path(shard.path).name
        observed_sha256, source_bytes = _download_verified_shard(shard, local_path)
        with (
            local_path.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as decompressed,
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
                if not spec.keeps_row(row):
                    filtered_rows += 1
                    continue
                sequence = validate_sequence(
                    row.get(sequence_field),
                    source=f"{stream}/{shard.path}:{line_number}",
                )
                uppercase_bases += sum(base.isupper() for base in sequence)
                lowercase_bases += sum(base.islower() for base in sequence)
                unknown_bases += sum(base not in CANONICAL_BASES for base in sequence)
                rows += 1
                yield {"text": {"content": sequence}}

    report = ShardReport(
        stream=stream,
        path=shard.path,
        source_sha256=observed_sha256,
        source_bytes=source_bytes,
        rows=rows,
        bases=rows * 255,
        filtered_rows=filtered_rows,
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
        partial(convert_shard, spec=spec, report_root=report_uri),
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
    filtered_rows = sum(int(report.get("filtered_rows", 0)) for report in reports)
    # A filtered stream keeps a subset, so the pinned count is the pre-filter
    # total; kept plus filtered must still account for every source row.
    if rows + filtered_rows != spec.expected_rows:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_rows} source rows, got "
            f"{rows} kept plus {filtered_rows} filtered"
        )
    manifest: dict[str, object] = {
        "format_version": 2,
        "stream": asdict(spec),
        "rows": rows,
        "filtered_rows": filtered_rows,
        "bases": sum(int(report["bases"]) for report in reports),
        "uppercase_bases": sum(int(report["uppercase_bases"]) for report in reports),
        "lowercase_bases": sum(int(report["lowercase_bases"]) for report in reports),
        "unknown_bases": sum(int(report["unknown_bases"]) for report in reports),
        "source_bytes": sum(int(report["source_bytes"]) for report in reports),
        "shards": reports,
    }
    _write_json(_join_uri(report_uri, "manifest.json"), manifest)
    return manifest
