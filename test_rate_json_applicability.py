import json
import shutil

from pathlib import Path

from src.models.document import Document
from src.models.rate import RateItem
from src.models.section import Section
from src.exporters.rate_json_exporter import (
    RateJSONExporter
)
from src.parsing.section_applicability_parser import (
    SectionApplicabilityParser
)
from src.pipelines.tariff_rate_pipeline import (
    TariffPipelineResult
)


def create_section(
    section_id: str,
    title: str,
    text: str,
    category: str
) -> Section:

    return Section(
        section_id=section_id,
        title=title,
        start_page=1,
        end_page=1,
        source_file="Test_Tariff.pdf",
        category=category,
        text=text
    )


def main():

    output_directory = Path(
        "output/test_rates"
    )

    if output_directory.exists():

        shutil.rmtree(
            output_directory
        )

    sections = [
        create_section(
            section_id="6.1.1.1.1",
            title="RESIDENTIAL SERVICE",
            category="NORMAL_SCHEDULE",
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
            category="TRANSITION_CHARGE",
            text=(
                "6.1.1.2.1 Rider TC - "
                "Transition Charge\n"
                "NOT APPLICABLE\n"
                "2"
            )
        ),
        create_section(
            section_id="6.1.1.3.1",
            title="EMPTY RIDER",
            category="RIDER",
            text=(
                "6.1.1.3.1 Empty Rider\n"
                "3"
            )
        )
    ]

    applicability_parser = (
        SectionApplicabilityParser()
    )

    applicability_results = (
        applicability_parser.parse_many(
            sections
        )
    )

    rate = RateItem(
        schedule_id="6.1.1.1.1",
        schedule_title="RESIDENTIAL SERVICE",
        category="NORMAL_SCHEDULE",
        source_file="Test_Tariff.pdf",
        charge_name="Customer Charge",
        value_text="$1.43",
        unit="per Retail Customer",
        source_method="TEXT",
        effective_date="May 1, 2023"
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
        text_rates=[rate],
        merged_rates=[rate],
        final_rates=[rate]
    )

    exporter = RateJSONExporter()

    exported_path = exporter.export_result(
        result=pipeline_result,
        output_directory=str(
            output_directory
        )
    )

    assert exported_path.exists()

    with exported_path.open(
        mode="r",
        encoding="utf-8"
    ) as input_file:

        payload = json.load(
            input_file
        )

    print("=" * 110)
    print("RATE JSON APPLICABILITY TEST")
    print("=" * 110)

    print(
        "Exported Path:",
        exported_path
    )

    print(
        "Schema Version:",
        payload["schema_version"]
    )

    print()

    for section in payload[
        "section_coverage"
    ]:

        print(
            section["section_id"],
            "|",
            section["title"]
        )

        print(
            "Applicability:",
            section[
                "applicability_status"
            ]
        )

        print(
            "Coverage:",
            section[
                "coverage_status"
            ]
        )

        print(
            "Rates:",
            section["rate_count"]
        )

        print("-" * 110)

    assert (
        payload["schema_version"]
        == "1.1"
    )

    coverage_summary = payload[
        "coverage_summary"
    ]

    assert (
        coverage_summary[
            "total_sections"
        ]
        == 3
    )

    assert (
        coverage_summary[
            "sections_with_rates"
        ]
        == 1
    )

    assert (
        coverage_summary[
            "not_applicable_sections"
        ]
        == 1
    )

    assert (
        coverage_summary[
            "review_required_sections"
        ]
        == 1
    )

    assert (
        len(
            payload[
                "not_applicable_sections"
            ]
        )
        == 1
    )

    assert (
        payload[
            "not_applicable_sections"
        ][0][
            "section_id"
        ]
        == "6.1.1.2.1"
    )

    assert (
        len(
            payload[
                "review_sections"
            ]
        )
        == 1
    )

    assert (
        payload[
            "review_sections"
        ][0][
            "section_id"
        ]
        == "6.1.1.3.1"
    )

    missing_section_ids = {
        section["section_id"]
        for section
        in payload["missing_sections"]
    }

    assert (
        "6.1.1.2.1"
        not in missing_section_ids
    )

    assert (
        "6.1.1.3.1"
        in missing_section_ids
    )

    print()
    print("=" * 110)
    print("ALL ASSERTIONS PASSED")
    print("=" * 110)


if __name__ == "__main__":
    main()