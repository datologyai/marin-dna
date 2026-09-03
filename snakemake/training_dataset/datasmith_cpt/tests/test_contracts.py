import pytest
from marin_dna_datasmith_cpt.cli import DEFAULT_ASSETS
from marin_dna_datasmith_cpt.contracts import (
    SEQUENCE_LENGTH,
    load_stream_specs,
    validate_sequence,
)


def test_checked_in_asset_manifest_pins_released_vertebrate_and_filtered_streams() -> None:
    assert DEFAULT_ASSETS.is_file()
    specs = load_stream_specs(DEFAULT_ASSETS)

    released = ("cds", "upstream", "downstream", "enhancer", "ncrna")
    vertebrate = (
        "vert_cds",
        "vert_tss_utr5",
        "vert_utr3",
        "vert_ncrna",
        "vert_ccre",
    )
    nohuman = tuple(f"{name}_nohuman" for name in vertebrate)
    filtered_families = tuple(
        f"{prefix}_{region}"
        for prefix in ("phylop", "gpnstar")
        for region in ("cds", "tss_utr5", "utr3", "ncrna", "ccre")
    )
    assert tuple(specs) == released + vertebrate + nohuman + filtered_families
    for name in filtered_families:
        assert specs[name].filter_drop == ("human_reference",)
    for name in nohuman:
        spec = specs[name]
        assert spec.filter_field == "alignment_source"
        assert spec.filter_drop == ("human_reference",)
        assert spec.expected_rows == specs[name.removesuffix("_nohuman")].expected_rows
    assert sum(
        spec.expected_rows for spec in specs.values() if not spec.is_filtered
    ) == 644_255_166
    assert all(spec.expected_shards == 64 for spec in specs.values())


def test_sequence_contract_preserves_case() -> None:
    sequence = "aNry" + "C" * (SEQUENCE_LENGTH - 4)

    assert validate_sequence(sequence, source="fixture") == sequence


@pytest.mark.parametrize(
    ("sequence", "message"),
    [
        ("A" * (SEQUENCE_LENGTH - 1), "sequence length"),
        ("A" * (SEQUENCE_LENGTH - 1) + "X", "invalid bases"),
        (None, "must be a string"),
    ],
)
def test_sequence_contract_rejects_invalid_rows(sequence: object, message: str) -> None:
    error = TypeError if sequence is None else ValueError
    with pytest.raises(error, match=message):
        validate_sequence(sequence, source="fixture")
