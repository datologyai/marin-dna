"""Command-line entrypoint for pinned MarinDNA CPT data conversion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from marin_dna_datasmith_cpt.contracts import load_stream_specs
from marin_dna_datasmith_cpt.conversion import (
    convert_stream,
    discover_shards,
    finalize_reports,
)
from marin_dna_datasmith_cpt.model_assets import validate_model_assets

DEFAULT_ASSETS = Path(__file__).parents[2] / "config" / "assets.toml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and convert pinned MarinDNA training streams"
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--stream", required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--stream", required=True)
    convert.add_argument("--output-uri", required=True)
    convert.add_argument("--report-uri", required=True)
    convert.add_argument("--num-workers", type=int, default=8)
    convert.add_argument("--chunk-bytes", default="64MB")
    convert.add_argument("--compression", choices=("zstd", "none"), default="zstd")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--stream", required=True)
    finalize.add_argument("--report-uri", required=True)

    validate_model = subparsers.add_parser("validate-model")
    validate_model.add_argument("--snapshot", type=Path, required=True)
    return parser


def main() -> None:
    """Run a discovery, conversion, or finalization command."""
    args = _parser().parse_args()
    if args.command == "validate-model":
        report = validate_model_assets(args.assets, args.snapshot)
        print(json.dumps(asdict(report), indent=2))
        return
    specs = load_stream_specs(args.assets)
    if args.stream not in specs:
        raise SystemExit(
            f"unknown stream {args.stream!r}; choose one of {sorted(specs)}"
        )
    spec = specs[args.stream]

    if args.command == "discover":
        print(json.dumps([asdict(shard) for shard in discover_shards(spec)], indent=2))
        return
    if args.command == "convert":
        if args.num_workers < 1:
            raise SystemExit("--num-workers must be positive")
        convert_stream(
            spec,
            output_uri=args.output_uri,
            report_uri=args.report_uri,
            num_workers=args.num_workers,
            chunk_bytes=args.chunk_bytes,
            compression=None if args.compression == "none" else args.compression,
        )
        finalize_reports(spec, report_uri=args.report_uri)
        return
    manifest = finalize_reports(spec, report_uri=args.report_uri)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
