from src.comparison.charge_identity_normalizer import (
    ChargeIdentityNormalizer
)
from src.models.rate import RateItem


SOURCE_FILE = "Test_Tariff.pdf"


def create_rate(
    schedule_title: str,
    charge_name: str
) -> RateItem:

    return RateItem(
        schedule_id="6.1.1.5.1",
        schedule_title=schedule_title,
        category="RIDER",
        source_file=SOURCE_FILE,
        charge_name=charge_name,
        value_text="$0.000000",
        unit="per kWh",
        source_method="DOCLING_ACCURATE",
        attributes={},
        metadata={}
    )


def main() -> None:

    normalizer = (
        ChargeIdentityNormalizer()
    )

    old_residential = create_rate(
        schedule_title=(
            "RIDER NDC - "
            "NUCLEAR DECOMMISSIONING CHARGES"
        ),
        charge_name=(
            "Stranded Cost Recovery Class - "
            "Nuclear Decommissioning Charge "
            "Factor (NDCF) - "
            "Residential Service"
        )
    )

    new_residential = create_rate(
        schedule_title=(
            "RIDER NDC - "
            "NUCLEAR DECOMMISSIONING CHARGES"
        ),
        charge_name=(
            "Rate Schedule - "
            "Nuclear Decommissioning Charge "
            "Factor (NDCF) - "
            "Residential Service"
        )
    )

    old_distribution = create_rate(
        schedule_title=(
            "RIDER NDC - "
            "NUCLEAR DECOMMISSIONING CHARGES"
        ),
        charge_name=(
            "Stranded Cost Recovery Class - "
            "Nuclear Decommissioning Charge "
            "Factor (NDCF) - "
            "Primary Service Greater than "
            "10 kW - Distribution Line"
        )
    )

    new_distribution = create_rate(
        schedule_title=(
            "RIDER NDC - "
            "NUCLEAR DECOMMISSIONING CHARGES"
        ),
        charge_name=(
            "Rate Schedule - "
            "Nuclear Decommissioning Charge "
            "Factor (NDCF) - "
            "Primary Service Greater than "
            "10 kW - Distribution Line"
        )
    )

    lighting = create_rate(
        schedule_title=(
            "RIDER NDC - "
            "NUCLEAR DECOMMISSIONING CHARGES"
        ),
        charge_name=(
            "Rate Schedule - "
            "Nuclear Decommissioning Charge "
            "Factor (NDCF) - "
            "Lighting Service"
        )
    )

    unrelated = create_rate(
        schedule_title="RESIDENTIAL SERVICE",
        charge_name="I. Customer Charge"
    )

    old_residential_identity = (
        normalizer.normalize(
            old_residential
        )
    )

    new_residential_identity = (
        normalizer.normalize(
            new_residential
        )
    )

    old_distribution_identity = (
        normalizer.normalize(
            old_distribution
        )
    )

    new_distribution_identity = (
        normalizer.normalize(
            new_distribution
        )
    )

    lighting_identity = (
        normalizer.normalize(
            lighting
        )
    )

    unrelated_identity = (
        normalizer.normalize(
            unrelated
        )
    )

    print("=" * 115)
    print("CHARGE IDENTITY NORMALIZER TEST")
    print("=" * 115)

    print()
    print(
        "Old Residential :",
        old_residential_identity
    )

    print(
        "New Residential :",
        new_residential_identity
    )

    print()
    print(
        "Old Distribution:",
        old_distribution_identity
    )

    print(
        "New Distribution:",
        new_distribution_identity
    )

    print()
    print(
        "Lighting        :",
        lighting_identity
    )

    print(
        "Unrelated       :",
        unrelated_identity
    )

    assert (
        old_residential_identity
        == "RESIDENTIAL SERVICE"
    )

    assert (
        new_residential_identity
        == "RESIDENTIAL SERVICE"
    )

    assert (
        old_residential_identity
        == new_residential_identity
    )

    assert (
        old_distribution_identity
        == (
            "PRIMARY SERVICE GREATER THAN "
            "10 KW - DISTRIBUTION LINE"
        )
    )

    assert (
        new_distribution_identity
        == old_distribution_identity
    )

    assert (
        lighting_identity
        == "LIGHTING SERVICE"
    )

    assert (
        unrelated_identity
        == "CUSTOMER CHARGE"
    )

    assert (
        lighting_identity
        != old_residential_identity
    )

    print()
    print("=" * 115)
    print("ALL ASSERTIONS PASSED")
    print("=" * 115)


if __name__ == "__main__":
    main()