import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_text(
    value: Any
) -> str:

    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\u2013",
        "-"
    )

    text = text.replace(
        "\u2014",
        "-"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().upper()


def normalize_value(
    rate: dict[str, Any]
) -> str:

    numeric_value = rate.get(
        "numeric_value"
    )

    if numeric_value is not None:
        return normalize_text(
            numeric_value
        )

    value_text = normalize_text(
        rate.get(
            "value_text",
            ""
        )
    )

    value_text = value_text.replace(
        "$",
        ""
    )

    value_text = value_text.replace(
        ",",
        ""
    )

    return value_text


def create_rate_identity(
    rate: dict[str, Any]
) -> tuple[str, ...]:
    """
    Creates a comparison identity that ignores effective date
    and extraction method.

    It is used only to determine whether an unresolved record
    has a matching dated record.
    """

    return (
        normalize_text(
            rate.get(
                "schedule_id",
                ""
            )
        ),
        normalize_text(
            rate.get(
                "normalized_charge_name",
                rate.get(
                    "charge_name",
                    ""
                )
            )
        ),
        normalize_text(
            rate.get(
                "normalized_unit",
                rate.get(
                    "unit",
                    ""
                )
            )
        ),
        normalize_value(
            rate
        )
    )


def get_metadata_value(
    rate: dict[str, Any],
    key: str
) -> str:

    metadata = rate.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict
    ):
        return ""

    return normalize_text(
        metadata.get(
            key,
            ""
        )
    )


def get_attribute_value(
    rate: dict[str, Any],
    key: str
) -> str:

    attributes = rate.get(
        "attributes",
        {}
    )

    if not isinstance(
        attributes,
        dict
    ):
        return ""

    return normalize_text(
        attributes.get(
            key,
            ""
        )
    )


def main():

    rates_directory = Path(
        "output/rates"
    )

    json_paths = sorted(
        rates_directory.glob(
            "*_rates.json"
        )
    )

    if not json_paths:

        raise FileNotFoundError(
            "No exported rate JSON files were found "
            "inside output/rates."
        )

    print("=" * 120)
    print("UNRESOLVED RATE DIAGNOSTICS")
    print("=" * 120)

    for json_path in json_paths:

        with json_path.open(
            mode="r",
            encoding="utf-8"
        ) as input_file:

            payload = json.load(
                input_file
            )

        source_file = payload[
            "source_document"
        ][
            "file_name"
        ]

        rates = payload.get(
            "rates",
            []
        )

        unresolved_rates = [
            rate
            for rate in rates
            if not normalize_text(
                rate.get(
                    "effective_date",
                    ""
                )
            )
        ]

        dated_rates = [
            rate
            for rate in rates
            if normalize_text(
                rate.get(
                    "effective_date",
                    ""
                )
            )
        ]

        dated_identities = {
            create_rate_identity(
                rate
            )
            for rate in dated_rates
        }

        unresolved_with_counterpart = [
            rate
            for rate in unresolved_rates
            if create_rate_identity(
                rate
            )
            in dated_identities
        ]

        unresolved_without_counterpart = [
            rate
            for rate in unresolved_rates
            if create_rate_identity(
                rate
            )
            not in dated_identities
        ]

        print()
        print("=" * 120)
        print(
            "SOURCE FILE:",
            source_file
        )
        print("=" * 120)

        print(
            "Total Rates                    :",
            len(rates)
        )

        print(
            "Dated Rates                    :",
            len(dated_rates)
        )

        print(
            "Unresolved Rates               :",
            len(unresolved_rates)
        )

        print(
            "With Matching Dated Record     :",
            len(
                unresolved_with_counterpart
            )
        )

        print(
            "Without Matching Dated Record  :",
            len(
                unresolved_without_counterpart
            )
        )

        print()
        print("UNRESOLVED COUNTS BY CATEGORY")
        print("-" * 120)

        category_counts = Counter(
            normalize_text(
                rate.get(
                    "category",
                    ""
                )
            )
            for rate in unresolved_rates
        )

        for category, count in (
            category_counts.most_common()
        ):

            print(
                f"{category:<40}:",
                count
            )

        print()
        print("UNRESOLVED COUNTS BY SOURCE METHOD")
        print("-" * 120)

        method_counts = Counter(
            normalize_text(
                rate.get(
                    "source_method",
                    ""
                )
            )
            for rate in unresolved_rates
        )

        for method, count in (
            method_counts.most_common()
        ):

            print(
                f"{method:<40}:",
                count
            )

        print()
        print("UNRESOLVED COUNTS BY TABLE STRUCTURE")
        print("-" * 120)

        structure_counts = Counter(
            get_metadata_value(
                rate,
                "table_structure"
            )
            or "NO_TABLE_STRUCTURE"
            for rate in unresolved_rates
        )

        for structure, count in (
            structure_counts.most_common()
        ):

            print(
                f"{structure:<40}:",
                count
            )

        unresolved_by_section = defaultdict(
            list
        )

        for rate in unresolved_rates:

            unresolved_by_section[
                (
                    rate.get(
                        "schedule_id",
                        ""
                    ),
                    rate.get(
                        "schedule_title",
                        ""
                    ),
                    rate.get(
                        "category",
                        ""
                    )
                )
            ].append(
                rate
            )

        print()
        print("UNRESOLVED DETAILS BY SECTION")
        print("=" * 120)

        ordered_sections = sorted(
            unresolved_by_section.items(),
            key=lambda item: (
                -len(item[1]),
                item[0][0]
            )
        )

        for (
            section_id,
            section_title,
            category
        ), section_rates in ordered_sections:

            section_with_counterpart = [
                rate
                for rate in section_rates
                if create_rate_identity(
                    rate
                )
                in dated_identities
            ]

            section_without_counterpart = [
                rate
                for rate in section_rates
                if create_rate_identity(
                    rate
                )
                not in dated_identities
            ]

            section_methods = Counter(
                normalize_text(
                    rate.get(
                        "source_method",
                        ""
                    )
                )
                for rate in section_rates
            )

            section_structures = Counter(
                get_metadata_value(
                    rate,
                    "table_structure"
                )
                or "NO_TABLE_STRUCTURE"
                for rate in section_rates
            )

            section_value_kinds = Counter(
                normalize_text(
                    rate.get(
                        "value_kind",
                        ""
                    )
                )
                for rate in section_rates
            )

            print()
            print("-" * 120)

            print(
                section_id,
                "|",
                section_title
            )

            print(
                "Category                  :",
                category
            )

            print(
                "Unresolved                :",
                len(section_rates)
            )

            print(
                "Matching Dated Counterpart:",
                len(
                    section_with_counterpart
                )
            )

            print(
                "No Dated Counterpart      :",
                len(
                    section_without_counterpart
                )
            )

            print(
                "Source Methods            :",
                dict(section_methods)
            )

            print(
                "Table Structures          :",
                dict(section_structures)
            )

            print(
                "Value Kinds               :",
                dict(section_value_kinds)
            )

            samples = (
                section_without_counterpart[
                    :8
                ]
            )

            if not samples:

                samples = section_rates[
                    :5
                ]

            print()
            print("Sample unresolved records:")

            for rate in samples:

                attributes = rate.get(
                    "attributes",
                    {}
                )

                if not isinstance(
                    attributes,
                    dict
                ):
                    attributes = {}

                print()
                print(
                    "  Charge :",
                    rate.get(
                        "charge_name",
                        ""
                    )
                )

                print(
                    "  Value  :",
                    rate.get(
                        "value_text",
                        ""
                    )
                )

                print(
                    "  Unit   :",
                    rate.get(
                        "unit",
                        ""
                    )
                )

                print(
                    "  Method :",
                    rate.get(
                        "source_method",
                        ""
                    )
                )

                print(
                    "  Page   :",
                    rate.get(
                        "page_number"
                    )
                )

                print(
                    "  Table  :",
                    rate.get(
                        "table_index"
                    )
                )

                print(
                    "  Row    :",
                    rate.get(
                        "row_index"
                    )
                )

                print(
                    "  Context:",
                    attributes.get(
                        "context_heading",
                        ""
                    )
                )

                print(
                    "  Row Label:",
                    attributes.get(
                        "row_label",
                        ""
                    )
                )

                print(
                    "  Header :",
                    attributes.get(
                        "column_header",
                        ""
                    )
                )

                print(
                    "  Has dated counterpart:",
                    (
                        create_rate_identity(
                            rate
                        )
                        in dated_identities
                    )
                )

        print()
        print("=" * 120)

    print()
    print("=" * 120)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()