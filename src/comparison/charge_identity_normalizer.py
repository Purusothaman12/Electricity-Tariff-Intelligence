import re

from typing import Any

from src.models.rate import RateItem


class ChargeIdentityNormalizer:
    """
    Produces stable charge identities for cross-document tariff
    comparison.

    The original charge name is never changed. This class returns
    a comparison-only identity.

    Example old NDC name:

        Stranded Cost Recovery Class -
        Nuclear Decommissioning Charge Factor (NDCF) -
        Residential Service$ 0.000218 per kWh

    Example new NDC name:

        Rate Schedule -
        Nuclear Decommissioning Charge Factor (NDCF) -
        Residential Service$ 0.000000

    Both become:

        RESIDENTIAL SERVICE
    """

    NDC_SCHEDULE_PATTERN = re.compile(
        r"\b(?:"
        r"NDC|"
        r"NUCLEAR\s+DECOMMISSIONING"
        r")\b",
        re.IGNORECASE
    )

    NDC_PREFIX_PATTERNS = (
        re.compile(
            r"^"
            r"STRANDED\s+COST\s+RECOVERY\s+CLASS"
            r"\s*-\s*"
            r"NUCLEAR\s+DECOMMISSIONING\s+"
            r"CHARGE\s+FACTOR"
            r"\s*\(NDCF\)"
            r"\s*-\s*",
            re.IGNORECASE
        ),
        re.compile(
            r"^"
            r"RATE\s+SCHEDULE"
            r"\s*-\s*"
            r"NUCLEAR\s+DECOMMISSIONING\s+"
            r"CHARGE\s+FACTOR"
            r"\s*\(NDCF\)"
            r"\s*-\s*",
            re.IGNORECASE
        ),
        re.compile(
            r"^"
            r"NUCLEAR\s+DECOMMISSIONING\s+"
            r"CHARGE\s+FACTOR"
            r"\s*\(NDCF\)"
            r"\s*-\s*",
            re.IGNORECASE
        )
    )

    LEADING_NUMBER_PATTERN = re.compile(
        r"^(?:"
        r"[IVXLCDM]+|"
        r"\d+(?:\.\d+)*"
        r")"
        r"[.)]?\s+",
        re.IGNORECASE
    )

    TRAILING_PUNCTUATION_PATTERN = re.compile(
        r"\s*[-:;,]+\s*$"
    )

    def normalize(
        self,
        rate: RateItem
    ) -> str:
        """
        Returns a stable comparison identity for one charge.
        """

        normalized_name = self._normalize_text(
            rate.normalized_charge_name
        )

        if not normalized_name:
            return ""

        normalized_name = (
            self.LEADING_NUMBER_PATTERN.sub(
                "",
                normalized_name
            )
        )

        normalized_name = self._normalize_text(
            normalized_name
        )

        if self._is_ndc_schedule(
            rate
        ):

            ndc_identity = (
                self._normalize_ndc_charge(
                    normalized_name
                )
            )

            if ndc_identity:
                return ndc_identity

        return normalized_name

    def _normalize_ndc_charge(
        self,
        charge_name: str
    ) -> str:
        """
        Removes known NDC prefixes and any rate value accidentally
        embedded in the charge name.
        """

        for pattern in self.NDC_PREFIX_PATTERNS:

            service_class = pattern.sub(
                "",
                charge_name,
                count=1
            )

            service_class = self._normalize_text(
                service_class
            )

            if (
                service_class
                and service_class != charge_name
            ):

                service_class = (
                    self._strip_embedded_value(
                        service_class
                    )
                )

                return service_class

        return self._strip_embedded_value(
            charge_name
        )

    def _strip_embedded_value(
        self,
        value: str
    ) -> str:
        """
        Removes an accidentally appended monetary rate.

        Example:

            RESIDENTIAL SERVICE$ 0.000218 PER KWH

        becomes:

            RESIDENTIAL SERVICE
        """

        cleaned_value = self._clean_text(
            value
        )

        if "$" in cleaned_value:

            cleaned_value = cleaned_value.split(
                "$",
                maxsplit=1
            )[0]

        cleaned_value = (
            self.TRAILING_PUNCTUATION_PATTERN.sub(
                "",
                cleaned_value
            )
        )

        return self._normalize_text(
            cleaned_value
        )

    def _is_ndc_schedule(
        self,
        rate: RateItem
    ) -> bool:

        schedule_text = " ".join(
            [
                self._clean_text(
                    rate.schedule_id
                ),
                self._clean_text(
                    rate.schedule_title
                )
            ]
        )

        return bool(
            self.NDC_SCHEDULE_PATTERN.search(
                schedule_text
            )
        )

    def _clean_text(
        self,
        value: Any
    ) -> str:

        if value is None:
            return ""

        text = str(
            value
        )

        text = text.replace(
            "\u00a0",
            " "
        )

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

        return text.strip()

    def _normalize_text(
        self,
        value: Any
    ) -> str:

        return self._clean_text(
            value
        ).upper()