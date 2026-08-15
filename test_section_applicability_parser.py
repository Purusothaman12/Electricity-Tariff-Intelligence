from src.models.section import Section
from src.parsing.section_applicability_parser import (
    SectionApplicability,
    SectionApplicabilityParser
)


def create_section(
    section_id: str,
    title: str,
    text: str
) -> Section:

    return Section(
        section_id=section_id,
        title=title,
        start_page=1,
        end_page=1,
        source_file="Test_Tariff.pdf",
        category="RIDER",
        text=text
    )


def main():

    parser = SectionApplicabilityParser()

    sections = [
        create_section(
            section_id="6.1.1.2.1",
            title="TC - TRANSITION CHARGE",
            text=(
                "6.1.1.2.1 Rider TC - "
                "Transition Charge\n"
                "NOT APPLICABLE\n"
                "91"
            )
        ),
        create_section(
            section_id="6.1.1.3.1",
            title=(
                "RIDER CTC - COMPETITION "
                "TRANSITION CHARGE"
            ),
            text=(
                "6.1.1.3.1 Rider CTC - "
                "Competition Transition Charge\n"
                "NOT CURRENTLY APPLICABLE.\n"
                "92"
            )
        ),
        create_section(
            section_id="6.1.1.1.1",
            title="RESIDENTIAL SERVICE",
            text=(
                "6.1.1.1.1 Residential Service\n"
                "Customer Charge $1.43 "
                "per Retail Customer\n"
                "Metering Charge $2.80 "
                "per Retail Customer\n"
                "67"
            )
        ),
        create_section(
            section_id="6.1.1.9.1",
            title="LEGAL CONDITIONS",
            text=(
                "6.1.1.9.1 Legal Conditions\n"
                "This provision is not applicable "
                "to customers receiving another "
                "service classification.\n"
                "110"
            )
        ),
        create_section(
            section_id="6.1.1.10.1",
            title="EMPTY SECTION",
            text=(
                "6.1.1.10.1 Empty Section\n"
                "120"
            )
        )
    ]

    results = parser.parse_many(
        sections
    )

    print("=" * 110)
    print("SECTION APPLICABILITY PARSER TEST")
    print("=" * 110)

    for result in results:

        print()
        print(
            "Section :",
            result.section_id,
            "|",
            result.section_title
        )

        print(
            "Status  :",
            result.status.value
        )

        print(
            "Reason  :",
            result.reason
        )

        print(
            "Matched :",
            result.matched_text
        )

        print(
            "Content :",
            result.substantive_lines
        )

        print("-" * 110)

    assert (
        results[0].status
        == SectionApplicability.NOT_APPLICABLE
    )

    assert (
        results[1].status
        == SectionApplicability.NOT_APPLICABLE
    )

    assert (
        results[2].status
        == SectionApplicability.APPLICABLE
    )

    assert (
        results[3].status
        == SectionApplicability.APPLICABLE
    )

    assert (
        results[4].status
        == SectionApplicability.UNKNOWN
    )

    print()
    print("=" * 110)
    print("ALL ASSERTIONS PASSED")
    print("=" * 110)


if __name__ == "__main__":
    main()