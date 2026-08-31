from __future__ import annotations

import hashlib
from pathlib import Path

import orjson
import pytest
import zstandard
from litdata import StreamingDataset
from marin_dna_datasmith_cpt.contracts import StreamSpec
from marin_dna_datasmith_cpt.conversion import (
    Shard,
    convert_shard,
    convert_stream,
    finalize_reports,
)


def _write_shard(path: Path, rows: list[dict[str, object]]) -> Shard:
    payload = b"".join(orjson.dumps(row) + b"\n" for row in rows)
    compressed = zstandard.ZstdCompressor().compress(payload)
    path.write_bytes(compressed)
    return Shard(
        path=f"data/train/{path.name}",
        url=str(path),
        size=len(compressed),
        sha256=hashlib.sha256(compressed).hexdigest(),
    )


def _spec(*, rows: int = 2) -> StreamSpec:
    return StreamSpec(
        name="fixture",
        repo_id="example/dataset",
        revision="a" * 40,
        sequence_field="seq",
        expected_rows=rows,
        expected_shards=1,
    )


def test_conversion_preserves_case_and_writes_a_verified_report(tmp_path: Path) -> None:
    sequences = ["A" * 255, "a" * 99 + "N" + "C" * 155]
    shard = _write_shard(
        tmp_path / "shard_0000.jsonl.zst",
        [
            {"id": str(index), "seq": sequence}
            for index, sequence in enumerate(sequences)
        ],
    )
    report_root = tmp_path / "reports"

    records = list(
        convert_shard(
            shard,
            stream="fixture",
            sequence_field="seq",
            report_root=str(report_root),
        )
    )

    assert [record["text"]["content"] for record in records] == sequences
    report = orjson.loads(
        (report_root / "shard_0000.jsonl.zst.report.json").read_bytes()
    )
    assert report["rows"] == 2
    assert report["bases"] == 510
    assert report["uppercase_bases"] == 411
    assert report["lowercase_bases"] == 99
    assert report["unknown_bases"] == 1
    assert report["source_sha256"] == shard.sha256

    manifest = finalize_reports(_spec(), report_uri=str(report_root))
    assert manifest["rows"] == 2
    assert manifest["bases"] == 510
    assert manifest["unknown_bases"] == 1
    assert (report_root / "manifest.json").is_file()


def test_conversion_rejects_invalid_rows_without_a_report(tmp_path: Path) -> None:
    shard = _write_shard(
        tmp_path / "shard_0000.jsonl.zst",
        [{"id": "bad", "seq": "A" * 254 + "X"}],
    )
    report_root = tmp_path / "reports"

    with pytest.raises(ValueError, match="invalid bases"):
        list(
            convert_shard(
                shard,
                stream="fixture",
                sequence_field="seq",
                report_root=str(report_root),
            )
        )

    assert not report_root.exists()


def test_conversion_rejects_compressed_hash_mismatch(tmp_path: Path) -> None:
    shard = _write_shard(
        tmp_path / "shard_0000.jsonl.zst",
        [{"id": "0", "seq": "A" * 255}],
    )
    wrong = Shard(path=shard.path, url=shard.url, size=shard.size, sha256="0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        list(
            convert_shard(
                wrong,
                stream="fixture",
                sequence_field="seq",
                report_root=str(tmp_path / "reports"),
            )
        )


def test_litdata_output_matches_universe_text_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = ["a" * 100 + "C" * 155, "G" * 255]
    shard = _write_shard(
        tmp_path / "shard_0000.jsonl.zst",
        [
            {"id": str(index), "seq": sequence}
            for index, sequence in enumerate(sequences)
        ],
    )
    output_root = tmp_path / "output.lit"

    monkeypatch.setattr(
        "marin_dna_datasmith_cpt.conversion.discover_shards", lambda spec: [shard]
    )
    convert_stream(
        _spec(),
        output_uri=str(output_root),
        report_uri=str(tmp_path / "reports"),
        num_workers=1,
        chunk_bytes="1MB",
        compression="zstd",
    )

    dataset = StreamingDataset(str(output_root), shuffle=False)
    assert [row["text"]["content"] for row in dataset] == sequences
