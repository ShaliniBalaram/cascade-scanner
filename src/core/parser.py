"""Natural language query parser."""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedQuery:
    location: str = "chennai"
    stakeholder: str = "emergency_manager"
    time_mode: str = "nowcast"
    asset_filter: Optional[str] = None
    confidence: float = 1.0


class QueryParser:
    """Parse natural language queries."""

    LOCATIONS = [(r"\b(chennai|madras)\b", "chennai")]
    TIMES = [
        (r"\b(now|current|today|latest)\b", "nowcast"),
        (r"\b(forecast|future|tomorrow)\b", "forecast"),
        (r"\b(past|historical|diagnostic)\b", "diagnostic"),
    ]
    STAKEHOLDERS = [
        (r"\b(emergency|ops|manager)\b", "emergency_manager"),
        (r"\b(research|scientist)\b", "researcher"),
    ]
    ASSETS = [
        (r"\b(hospital|medical)\b", "hospital"),
        (r"\b(substation|power)\b", "substation"),
        (r"\b(road|highway)\b", "evacuation_route"),
    ]

    def parse(self, query: str) -> ParsedQuery:
        if not query:
            return ParsedQuery(confidence=0.5)

        q = query.lower()
        result = ParsedQuery()

        result.location = self._match(q, self.LOCATIONS, "chennai")
        result.time_mode = self._match(q, self.TIMES, "nowcast")
        result.stakeholder = self._match(q, self.STAKEHOLDERS, "emergency_manager")
        result.asset_filter = self._match(q, self.ASSETS, None)

        return result

    def _match(self, text: str, patterns: list, default):
        for pattern, value in patterns:
            if re.search(pattern, text, re.I):
                return value
        return default


query_parser = QueryParser()
