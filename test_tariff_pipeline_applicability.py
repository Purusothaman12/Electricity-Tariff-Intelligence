from src.models.document import Document
from src.models.section import Section
from src.pipelines.tariff_rate_pipeline import (
    TariffPipelineResult,
    TariffRatePipeline
)
from src.parsing.section_applicability_parser import (
    SectionApplicability
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

    pipeline = TariffRatePipeline(
        max_pages_per_batch=3
    )

    sections = [
        create_section(
            section_id="6.1.1.1.1",
            title="RESIDENTIAL SERVICE",
            text=(
                "6.1.1.1.1 Residential Service\n"
                "Customer Charge $1.43 "
                "per Retail Customer\n"
                "1"
            )
        ),
        create_section(
            section_id="6.1.1.2.1",
            title="TC - TRANSITION CHARGE",
            text=(
                "6.1.1.2.1 Rider TC - "
                "Transition Charge\n"
                "NOT APPLICABLE\n"
                "2"
            )
        ),
        create_section(
            section_id="6.1.1.3.1",
            title="EMPTY SECTION",
            text=(
                "6.1.1.3.1 Empty Section\n"
                "3"
            )
        )
    ]

    applicability_results = (
        pipeline
        .section_applicability_parser
        .parse_many(
            sections
        )
    )

    processable_sections = (
        pipeline.get_processable_sections(
            sections=sections,
            applicability_results=(
                applicability_results
            )
        )
    )

    processable_ids = {
        section.section_id
        for section in processable_sections
    }

    print("=" * 110)
    print("TARIFF PIPELINE APPLICABILITY TEST")
    print("=" * 110)

    for result in applicability_results:

        print()
        print(
            "Section:",
            result.section_id,
            "|",
            result.section_title
        )

        print(
            "Status :",
            result.status.value
        )

        print(
            "Sent for extraction:",
            (
                result.section_id
                in processable_ids
            )
        )

        print("-" * 110)

    assert (
        "6.1.1.1.1"
        in processable_ids
    )

    assert (
        "6.1.1.2.1"
        not in processable_ids
    )

    assert (
        "6.1.1.3.1"
        in processable_ids
    )

    document = Document(
        file_name="Test_Tariff.pdf",
        file_path="data/Test_Tariff.pdf",
        pages=[]
    )

    pipeline_result = TariffPipelineResult(
        document=document,
        sections=sections,
        section_applicability=(
            applicability_results
        ),
        tables=[],
        table_rates=[],
        text_rates=[],
        merged_rates=[],
        final_rates=[]
    )

    summary = pipeline_result.to_summary()

    print()
    print("PIPELINE SUMMARY")
    print("-" * 110)

    for key, value in summary.items():

        print(
            f"{key:<25}:",
            value
        )

    assert (
        pipeline_result
        .applicable_section_count
        == 1
    )

    assert (
        pipeline_result
        .not_applicable_section_count
        == 1
    )

    assert (
        pipeline_result
        .unknown_section_count
        == 1
    )

    assert (
        pipeline_result
        .processable_section_count
        == 2
    )

    assert (
        pipeline_result
        .not_applicable_section_ids
        == ["6.1.1.2.1"]
    )

    not_applicable_result = (
        pipeline_result.get_applicability(
            "6.1.1.2.1"
        )
    )

    assert not_applicable_result is not None

    assert (
        not_applicable_result.status
        == SectionApplicability
        .NOT_APPLICABLE
    )

    print()
    print("=" * 110)
    print("ALL ASSERTIONS PASSED")
    print("=" * 110)


if __name__ == "__main__":
    main()