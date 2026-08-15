from collections import Counter
from collections import defaultdict

from src.comparison.rate_comparator import (
    RateChangeStatus,
    RateComparator,
    RateComparisonRecord
)
from src.loaders.rate_json_loader import (
    RateJSONLoader
)


OLD_SOURCE_FILE = (
    "Oncor_November_27_2017.pdf"
)

NEW_SOURCE_FILE = (
    "Oncor_May_1_2023.pdf"
)

MAX_SCHEDULES_TO_PRINT = 30
MAX_CANDIDATES_TO_PRINT = 40
MAX_SAMPLES_PER_SCHEDULE = 5


def get_side_rate(
    comparison: RateComparisonRecord,
    side: str
):

    if side == "OLD":
        return comparison.old_rate

    if side == "NEW":
        return comparison.new_rate

    raise ValueError(
        f"Unsupported side: {side}"
    )


def get_schedule_label(
    comparison: RateComparisonRecord,
    side: str
) -> str:

    rate = get_side_rate(
        comparison,
        side
    )

    if rate is None:
        return "UNKNOWN"

    return (
        f"{rate.schedule_id} | "
        f"{rate.schedule_title}"
    )


def get_category(
    comparison: RateComparisonRecord,
    side: str
) -> str:

    rate = get_side_rate(
        comparison,
        side
    )

    if rate is None:
        return "UNKNOWN"

    return rate.category


def identity_without_schedule(
    comparison: RateComparisonRecord
) -> tuple[str, ...]:
    """
    Removes only the schedule-title component.

    Matching records here probably differ because the Rider or
    schedule title changed between tariff documents.
    """

    return comparison.identity[1:]


def identity_without_context(
    comparison: RateComparisonRecord
) -> tuple[str, ...]:
    """
    Removes the normal text-context component while retaining
    schedule, category, charge, unit and matrix dimensions.
    """

    identity = comparison.identity

    return (
        identity[0],
        identity[1],
        identity[2],
        identity[3],
        identity[5],
        identity[6]
    )


def identity_without_unit(
    comparison: RateComparisonRecord
) -> tuple[str, ...]:
    """
    Removes only the unit component.
    """

    identity = comparison.identity

    return (
        identity[0],
        identity[1],
        identity[2],
        identity[4],
        identity[5],
        identity[6]
    )


def print_unmatched_schedule_summary(
    title: str,
    comparisons: list[RateComparisonRecord],
    side: str
) -> None:

    schedule_counts = Counter(
        get_schedule_label(
            comparison,
            side
        )
        for comparison in comparisons
    )

    print()
    print(title)
    print("-" * 120)

    for schedule, count in (
        schedule_counts.most_common(
            MAX_SCHEDULES_TO_PRINT
        )
    ):

        print(
            f"{count:>5} | {schedule}"
        )


def print_unmatched_category_summary(
    title: str,
    comparisons: list[RateComparisonRecord],
    side: str
) -> None:

    category_counts = Counter(
        get_category(
            comparison,
            side
        )
        for comparison in comparisons
    )

    print()
    print(title)
    print("-" * 120)

    for category, count in (
        category_counts.most_common()
    ):

        print(
            f"{category:<30}: {count}"
        )


def print_schedule_samples(
    title: str,
    comparisons: list[RateComparisonRecord],
    side: str
) -> None:

    grouped = defaultdict(
        list
    )

    for comparison in comparisons:

        grouped[
            get_schedule_label(
                comparison,
                side
            )
        ].append(
            comparison
        )

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    )

    print()
    print(title)
    print("=" * 120)

    for schedule, schedule_records in (
        ordered_groups[
            :MAX_SCHEDULES_TO_PRINT
        ]
    ):

        print()
        print("-" * 120)

        print(
            f"{schedule} | "
            f"Count: {len(schedule_records)}"
        )

        for comparison in (
            schedule_records[
                :MAX_SAMPLES_PER_SCHEDULE
            ]
        ):

            rate = get_side_rate(
                comparison,
                side
            )

            if rate is None:
                continue

            print(
                f"  {rate.charge_name:<55}"
                f"{rate.value_text:<20}"
                f"{comparison.unit:<35}"
                f"{comparison.context}"
            )


def build_candidate_pairs(
    removed_records: list[
        RateComparisonRecord
    ],
    added_records: list[
        RateComparisonRecord
    ],
    key_function
) -> list[
    tuple[
        RateComparisonRecord,
        RateComparisonRecord
    ]
]:

    removed_by_key = defaultdict(
        list
    )

    added_by_key = defaultdict(
        list
    )

    for comparison in removed_records:

        removed_by_key[
            key_function(
                comparison
            )
        ].append(
            comparison
        )

    for comparison in added_records:

        added_by_key[
            key_function(
                comparison
            )
        ].append(
            comparison
        )

    common_keys = (
        set(removed_by_key)
        & set(added_by_key)
    )

    candidate_pairs = []

    for key in sorted(
        common_keys
    ):

        old_candidates = (
            removed_by_key[key]
        )

        new_candidates = (
            added_by_key[key]
        )

        for old_comparison in (
            old_candidates
        ):

            for new_comparison in (
                new_candidates
            ):

                candidate_pairs.append(
                    (
                        old_comparison,
                        new_comparison
                    )
                )

    return candidate_pairs


def print_candidate_pairs(
    title: str,
    explanation: str,
    candidate_pairs: list[
        tuple[
            RateComparisonRecord,
            RateComparisonRecord
        ]
    ]
) -> None:

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)

    print(explanation)

    print(
        "Candidate pair count:",
        len(candidate_pairs)
    )

    for old_comparison, new_comparison in (
        candidate_pairs[
            :MAX_CANDIDATES_TO_PRINT
        ]
    ):

        old_rate = (
            old_comparison.old_rate
        )

        new_rate = (
            new_comparison.new_rate
        )

        if (
            old_rate is None
            or new_rate is None
        ):
            continue

        print()
        print("-" * 120)

        print(
            "Old Schedule :",
            old_rate.schedule_id,
            "|",
            old_rate.schedule_title
        )

        print(
            "New Schedule :",
            new_rate.schedule_id,
            "|",
            new_rate.schedule_title
        )

        print(
            "Old Charge   :",
            old_rate.charge_name
        )

        print(
            "New Charge   :",
            new_rate.charge_name
        )

        print(
            "Old Value    :",
            old_rate.value_text
        )

        print(
            "New Value    :",
            new_rate.value_text
        )

        print(
            "Old Unit     :",
            old_comparison.unit
        )

        print(
            "New Unit     :",
            new_comparison.unit
        )

        print(
            "Old Context  :",
            old_comparison.context
        )

        print(
            "New Context  :",
            new_comparison.context
        )

        print(
            "Old Identity :",
            old_comparison.identity
        )

        print(
            "New Identity :",
            new_comparison.identity
        )


def main() -> None:

    loader = RateJSONLoader()

    documents = loader.load_directory(
        "output/rates"
    )

    documents_by_source = {
        document.source_file: document
        for document in documents
    }

    old_document = documents_by_source[
        OLD_SOURCE_FILE
    ]

    new_document = documents_by_source[
        NEW_SOURCE_FILE
    ]

    comparator = RateComparator()

    result = comparator.compare(
        old_document=old_document,
        new_document=new_document
    )

    added_records = [
        comparison
        for comparison
        in result.comparisons
        if (
            comparison.status
            == RateChangeStatus.ADDED
        )
    ]

    removed_records = [
        comparison
        for comparison
        in result.comparisons
        if (
            comparison.status
            == RateChangeStatus.REMOVED
        )
    ]

    matched_records = [
        comparison
        for comparison
        in result.comparisons
        if (
            comparison.old_rate
            is not None
            and comparison.new_rate
            is not None
        )
    ]

    print("=" * 120)
    print(
        "UNMATCHED COMPARISON DIAGNOSTIC"
    )
    print("=" * 120)

    print()
    print("SUMMARY")
    print("-" * 120)

    print(
        "Old snapshot records :",
        result.old_snapshot_count
    )

    print(
        "New snapshot records :",
        result.new_snapshot_count
    )

    print(
        "Matched records      :",
        len(matched_records)
    )

    print(
        "Added records        :",
        len(added_records)
    )

    print(
        "Removed records      :",
        len(removed_records)
    )

    old_match_ratio = (
        len(matched_records)
        / result.old_snapshot_count
        * 100
    )

    new_match_ratio = (
        len(matched_records)
        / result.new_snapshot_count
        * 100
    )

    print(
        "Old match coverage   :",
        f"{old_match_ratio:.2f}%"
    )

    print(
        "New match coverage   :",
        f"{new_match_ratio:.2f}%"
    )

    print_unmatched_category_summary(
        title=(
            "REMOVED RECORDS BY CATEGORY"
        ),
        comparisons=removed_records,
        side="OLD"
    )

    print_unmatched_category_summary(
        title=(
            "ADDED RECORDS BY CATEGORY"
        ),
        comparisons=added_records,
        side="NEW"
    )

    print_unmatched_schedule_summary(
        title=(
            "REMOVED RECORDS BY OLD SCHEDULE"
        ),
        comparisons=removed_records,
        side="OLD"
    )

    print_unmatched_schedule_summary(
        title=(
            "ADDED RECORDS BY NEW SCHEDULE"
        ),
        comparisons=added_records,
        side="NEW"
    )

    schedule_title_candidates = (
        build_candidate_pairs(
            removed_records=removed_records,
            added_records=added_records,
            key_function=(
                identity_without_schedule
            )
        )
    )

    context_candidates = (
        build_candidate_pairs(
            removed_records=removed_records,
            added_records=added_records,
            key_function=(
                identity_without_context
            )
        )
    )

    unit_candidates = (
        build_candidate_pairs(
            removed_records=removed_records,
            added_records=added_records,
            key_function=(
                identity_without_unit
            )
        )
    )

    print_candidate_pairs(
        title=(
            "POSSIBLE SCHEDULE-TITLE MISMATCHES"
        ),
        explanation=(
            "These old and new records match after "
            "removing only the schedule title from "
            "their comparison identities."
        ),
        candidate_pairs=(
            schedule_title_candidates
        )
    )

    print_candidate_pairs(
        title=(
            "POSSIBLE CONTEXT MISMATCHES"
        ),
        explanation=(
            "These records match after removing "
            "only their text-context component."
        ),
        candidate_pairs=(
            context_candidates
        )
    )

    print_candidate_pairs(
        title=(
            "POSSIBLE UNIT MISMATCHES"
        ),
        explanation=(
            "These records match after removing "
            "only their normalized unit."
        ),
        candidate_pairs=(
            unit_candidates
        )
    )

    print_schedule_samples(
        title=(
            "SAMPLE REMOVED RECORDS"
        ),
        comparisons=removed_records,
        side="OLD"
    )

    print_schedule_samples(
        title=(
            "SAMPLE ADDED RECORDS"
        ),
        comparisons=added_records,
        side="NEW"
    )

    assert (
        len(matched_records)
        + len(removed_records)
        == result.old_snapshot_count
    )

    assert (
        len(matched_records)
        + len(added_records)
        == result.new_snapshot_count
    )

    assert (
        len(result.comparisons)
        == (
            len(matched_records)
            + len(removed_records)
            + len(added_records)
        )
    )

    print()
    print("=" * 120)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()