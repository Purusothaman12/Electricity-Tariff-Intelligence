import re

from src.models.section import TOCEntry


class ScheduleClassifier:
    """
    Classifies tariff TOC entries into schedule categories.
    """

    NORMAL_SCHEDULE = "NORMAL_SCHEDULE"
    RIDER = "RIDER"
    TRANSITION_CHARGE = "TRANSITION_CHARGE"
    UNKNOWN = "UNKNOWN"

    def classify(
        self,
        entry: TOCEntry
    ) -> str:

        title = self._normalize_title(
            entry.title
        )

        if title.startswith("RIDER "):
            return self.RIDER

        if (
            title == "TC - TRANSITION CHARGE"
            or "TRANSITION CHARGE" in title
        ):
            return self.TRANSITION_CHARGE

        normal_schedule_names = {
            "RESIDENTIAL SERVICE",
            "SECONDARY SERVICE LESS THAN OR EQUAL TO 10 KW",
            "SECONDARY SERVICE GREATER THAN 10 KW",
            "PRIMARY SERVICE LESS THAN OR EQUAL TO 10 KW",
            (
                "PRIMARY SERVICE GREATER THAN 10 KW "
                "- DISTRIBUTION LINE"
            ),
            (
                "PRIMARY SERVICE GREATER THAN 10 KW "
                "- SUBSTATION"
            ),
            "TRANSMISSION SERVICE",
            "LIGHTING SERVICE"
        }

        if title in normal_schedule_names:
            return self.NORMAL_SCHEDULE

        return self.UNKNOWN

    def classify_all(
        self,
        entries: list[TOCEntry]
    ) -> dict[str, list[TOCEntry]]:

        classified = {
            self.NORMAL_SCHEDULE: [],
            self.RIDER: [],
            self.TRANSITION_CHARGE: [],
            self.UNKNOWN: []
        }

        for entry in entries:

            category = self.classify(
                entry
            )

            classified[category].append(
                entry
            )

        return classified

    def normalize_schedule_name(
        self,
        title: str
    ) -> str:

        title = self._normalize_title(
            title
        )

        title = re.sub(
            r"^RIDER\s+",
            "",
            title
        )

        title = re.sub(
            r"\s*\([^)]*\)\s*$",
            "",
            title
        )

        title = re.sub(
            r"\s*-\s*",
            " - ",
            title
        )

        title = re.sub(
            r"\s+",
            " ",
            title
        )

        return title.strip()

    def _normalize_title(
        self,
        title: str
    ) -> str:

        title = title.upper()

        title = title.replace(
            "\u2013",
            "-"
        )

        title = title.replace(
            "\u2014",
            "-"
        )

        title = re.sub(
            r"\s+",
            " ",
            title
        )

        return title.strip()