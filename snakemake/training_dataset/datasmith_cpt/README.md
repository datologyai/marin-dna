# DataSmith CPT onboarding

This project converts the five immutable public MarinDNA m5.1 training streams, plus the five pinned `marin-dna/vertebrate-v1-*` continuation streams, into the raw-string LitData format consumed by the Universe TorchTitan/Zephon training path.
It is an onboarding adapter, not a replacement for the original training-dataset generation workflows.

The checked-in [`config/assets.toml`](config/assets.toml) pins every public repository revision and expected row count.
Every source shard is streamed once, validated, and SHA-256 checked against its Hugging Face LFS object ID while its records are written as `{"text": {"content": <255 mixed-case bases>}}`.
Original case is preserved for the reproduction-grade weighted objective.

## Contract

- exactly 64 contiguous shards per stream;
- exactly 255 bases per record;
- alphabet restricted to standard mixed-case IUPAC DNA symbols;
- ambiguous bases are preserved and map to the released tokenizer's `[UNK]` token;
- source compressed size and SHA-256 must match the pinned inventory;
- final row count must match the checked-in stream total;
- a report is written only after a whole shard passes validation;
- the aggregate manifest is written only after the complete stream passes.

## Usage

Inspect the pinned inventory without downloading the shards:

```bash
uv run --locked marin-dna-cpt-data discover --stream cds
```

Validate a downloaded m5.1 snapshot before mirroring or training:

```bash
uv run --locked marin-dna-cpt-data validate-model --snapshot /path/to/snapshot
```

Convert one stream to a local or S3 LitData dataset:

```bash
uv run --locked marin-dna-cpt-data convert \
  --stream cds \
  --output-uri s3://BUCKET/PREFIX/cds/data.lit \
  --report-uri s3://BUCKET/PREFIX/cds/reports \
  --num-workers 8
```

The command fails instead of silently dropping invalid rows.
Production execution should use an immutable image built from this project and a fresh output prefix.

Build the project image from this directory:

```bash
docker build --platform linux/amd64 -t marin-dna-cpt-data .
docker run --rm marin-dna-cpt-data discover --stream cds
```

## Verification

```bash
uv sync --locked --group dev
uv run --locked pytest
```
