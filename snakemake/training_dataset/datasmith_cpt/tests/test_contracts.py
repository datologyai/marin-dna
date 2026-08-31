import pytest
from marin_dna_datasmith_cpt.cli import DEFAULT_ASSETS
from marin_dna_datasmith_cpt.contracts import (
    SEQUENCE_LENGTH,
    load_stream_specs,
    validate_sequence,
)

def test_checked_in_asset_manifest_pins_five_streams() -> None:
    assert DEFAULT_ASSETS.is_file()
    specs = load_stream_specs(DEFAULT_ASSETS)

    assert tuple(specs) == ("cds", "upstream", "downstream", "enhancer", "ncrna")
    assert sum(spec.expected_rows for spec in specs.values()) == 443_039_602
    assert all(spec.expected_shards == 64 for spec in specs.values())


def test_sequence_contract_preserves_case() -> None:
    sequence = "a" + "C" * (SEQUENCE_LENGTH - 1)

    assert validate_sequence(sequence, source="fixture") == sequence


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("A" * (SEQUENCE_LENGTH - 1), "sequence length"),
        ("A" * (SEQUENCE_LENGTH - 1) + "N", "invalid bases"),
        (None, "must be a string"),
    ],
)
def test_sequence_contract_rejects_invalid_rows(sequence: object, message: str) -> None:
    error = TypeError if sequence is None else ValueError
    with pytest.raises(error, match=message):
        validate_sequence(sequence, source="fixture")
