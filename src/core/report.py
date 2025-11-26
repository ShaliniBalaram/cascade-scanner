"""Report generator with data source attribution."""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import requests

from loguru import logger


@dataclass
class DataSource:
    name: str
    provider: str
    url: str
    data_type: str
    latency: str  # How fresh is the data
    coverage: str


# Data source registry
DATA_SOURCES = {
    "precipitation": DataSource(
        name="Open-Meteo Weather API",
        provider="Open-Meteo (ERA5 reanalysis + ECMWF)",
        url="https://open-meteo.com/",
        data_type="Precipitation (mm)",
        latency="Archive: 5-7 days delay, Forecast API: real-time",
        coverage="Global, hourly/daily resolution",
    ),
    "satellite_sar": DataSource(
        name="Sentinel-1 SAR GRD",
        provider="ESA Copernicus via Google Earth Engine",
        url="https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD",
        data_type="C-band SAR backscatter (VV/VH polarization)",
        latency="6-12 day revisit cycle + 1-3 day processing",
        coverage="Chennai region (12.85-13.25°N, 80.10-80.35°E)",
    ),
    "imd_official": DataSource(
        name="India Meteorological Department",
        provider="IMD (Government of India)",
        url="https://mausam.imd.gov.in/",
        data_type="Official rainfall measurements",
        latency="Real-time for alerts, daily for archives",
        coverage="India - station-based observations",
    ),
}


@dataclass
class ReportResult:
    answer: str
    data_source: DataSource
    raw_data: dict
    confidence: str
    timestamp: datetime


class ReportEngine:
    """Answer questions about weather/hazard data with attribution."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.precip_cache = None
        self._load_local_data()

    def _load_local_data(self):
        """Load downloaded precipitation data."""
        precip_dir = self.data_dir / "precipitation"
        if precip_dir.exists():
            csv_files = list(precip_dir.glob("*_daily.csv"))
            if csv_files:
                self.precip_cache = self._parse_csv(csv_files[0])
                logger.info(f"Loaded {len(self.precip_cache)} days of precip data")

    def _parse_csv(self, filepath: Path) -> dict:
        """Parse precipitation CSV into dict."""
        data = {}
        with open(filepath) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    date = parts[0]
                    precip = float(parts[1]) if parts[1] else 0
                    data[date] = {"precipitation_mm": precip, "rain_mm": float(parts[2]) if len(parts) > 2 else precip}
        return data

    def query(self, question: str) -> ReportResult:
        """Answer a natural language question about weather/hazards."""
        q = question.lower()

        # Check for data source info request
        if any(w in q for w in ["data source", "sources", "where does", "what data"]):
            return ReportResult(
                answer=self.get_data_sources_info(),
                data_source=DATA_SOURCES["precipitation"],
                raw_data={"sources": list(DATA_SOURCES.keys())},
                confidence="high",
                timestamp=datetime.now(),
            )

        # Parse date references
        target_date = self._parse_date_reference(q)

        # Determine query type
        if any(w in q for w in ["rain", "rainfall", "precipitation", "precip"]):
            return self._query_precipitation(q, target_date)
        elif any(w in q for w in ["flood", "water", "inundation", "sar", "satellite"]):
            return self._query_satellite(q, target_date)
        else:
            return self._query_precipitation(q, target_date)  # default

    def _parse_date_reference(self, text: str) -> Optional[datetime]:
        """Extract date from natural language."""
        today = datetime.now()

        # ISO date format: 2025-11-18
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if date_match:
            return datetime.strptime(date_match.group(1), "%Y-%m-%d")

        # Month day format: "November 18" or "Nov 18"
        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
            "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
        }
        for month_name, month_num in months.items():
            pattern = rf"{month_name}\s+(\d{{1,2}})"
            match = re.search(pattern, text.lower())
            if match:
                day = int(match.group(1))
                year = today.year
                try:
                    return datetime(year, month_num, day)
                except ValueError:
                    pass

        # Relative references
        if "today" in text:
            return today
        if "yesterday" in text:
            return today - timedelta(days=1)

        # Day of week references
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(days):
            if day in text:
                # Find the most recent occurrence
                current_dow = today.weekday()
                days_ago = (current_dow - i) % 7
                if days_ago == 0 and "last" in text:
                    days_ago = 7
                if "last week" in text:
                    days_ago += 7
                return today - timedelta(days=days_ago)

        # "last X days"
        days_match = re.search(r"last (\d+) days?", text)
        if days_match:
            return today - timedelta(days=int(days_match.group(1)))

        return today

    def _query_precipitation(self, question: str, target_date: Optional[datetime]) -> ReportResult:
        """Query precipitation data."""
        source = DATA_SOURCES["precipitation"]

        # Try local cache first
        if self.precip_cache and target_date:
            date_str = target_date.strftime("%Y-%m-%d")
            if date_str in self.precip_cache:
                data = self.precip_cache[date_str]
                precip = data["precipitation_mm"]

                if precip > 0:
                    answer = f"Yes, there was rainfall on {target_date.strftime('%A, %B %d, %Y')}. "\
                             f"Recorded precipitation: **{precip:.1f} mm**."
                    if precip > 50:
                        answer += " This was heavy rainfall (>50mm)."
                    elif precip > 20:
                        answer += " This was moderate rainfall."
                    else:
                        answer += " This was light rainfall."
                else:
                    answer = f"No rainfall was recorded on {target_date.strftime('%A, %B %d, %Y')} (0.0 mm)."

                return ReportResult(
                    answer=answer,
                    data_source=source,
                    raw_data={"date": date_str, **data},
                    confidence="high" if abs((datetime.now() - target_date).days) < 30 else "medium",
                    timestamp=datetime.now(),
                )

        # Fetch from API if not in cache
        return self._fetch_precipitation_api(target_date or datetime.now(), source)

    def _fetch_precipitation_api(self, target_date: datetime, source: DataSource) -> ReportResult:
        """Fetch precipitation from Open-Meteo API."""
        lat, lon = 13.0827, 80.2707

        # Use forecast API for recent dates (has past_days parameter)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,rain_sum",
            "past_days": 14,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])

            date_str = target_date.strftime("%Y-%m-%d")
            if date_str in dates:
                idx = dates.index(date_str)
                mm = precip[idx] or 0

                if mm > 0:
                    answer = f"Yes, rainfall on {target_date.strftime('%A, %B %d')}: **{mm:.1f} mm**."
                else:
                    answer = f"No rainfall on {target_date.strftime('%A, %B %d')} (0.0 mm)."

                return ReportResult(
                    answer=answer,
                    data_source=source,
                    raw_data={"date": date_str, "precipitation_mm": mm},
                    confidence="high",
                    timestamp=datetime.now(),
                )
            else:
                return ReportResult(
                    answer=f"Data not available for {date_str}. Available range: {dates[0]} to {dates[-1]}.",
                    data_source=source,
                    raw_data={"available_dates": dates},
                    confidence="low",
                    timestamp=datetime.now(),
                )

        except Exception as e:
            return ReportResult(
                answer=f"Could not fetch data: {e}",
                data_source=source,
                raw_data={"error": str(e)},
                confidence="none",
                timestamp=datetime.now(),
            )

    def _query_satellite(self, question: str, target_date: Optional[datetime]) -> ReportResult:
        """Query satellite data availability."""
        source = DATA_SOURCES["satellite_sar"]

        # Check local SAR files
        sar_dir = self.data_dir / "sentinel1"
        sar_files = list(sar_dir.glob("*.tif")) if sar_dir.exists() else []

        if sar_files:
            latest = max(sar_files, key=lambda f: f.stat().st_mtime)
            # Extract date from filename
            date_match = re.search(r"(\d{8})", latest.name)
            if date_match:
                img_date = datetime.strptime(date_match.group(1), "%Y%m%d")
                days_old = (datetime.now() - img_date).days

                answer = f"Latest Sentinel-1 SAR image: **{img_date.strftime('%B %d, %Y')}** ({days_old} days old). "\
                         f"File: {latest.name}. "\
                         f"Note: Sentinel-1 has 6-12 day revisit cycle."

                return ReportResult(
                    answer=answer,
                    data_source=source,
                    raw_data={"file": str(latest), "date": img_date.isoformat(), "days_old": days_old},
                    confidence="high",
                    timestamp=datetime.now(),
                )

        return ReportResult(
            answer="No local Sentinel-1 data. Run `python scripts/download_data.py` to download.",
            data_source=source,
            raw_data={},
            confidence="none",
            timestamp=datetime.now(),
        )

    def get_data_sources_info(self) -> str:
        """Return info about all data sources."""
        lines = ["## Data Sources Used\n"]
        for key, src in DATA_SOURCES.items():
            lines.append(f"### {src.name}")
            lines.append(f"- **Provider:** {src.provider}")
            lines.append(f"- **Data Type:** {src.data_type}")
            lines.append(f"- **Latency:** {src.latency}")
            lines.append(f"- **Coverage:** {src.coverage}")
            lines.append(f"- **URL:** {src.url}")
            lines.append("")
        return "\n".join(lines)

    def format_report(self, result: ReportResult) -> str:
        """Format a report result with attribution."""
        lines = [
            result.answer,
            "",
            "---",
            f"**Data Source:** {result.data_source.name}",
            f"**Provider:** {result.data_source.provider}",
            f"**Confidence:** {result.confidence}",
            f"**Query Time:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return "\n".join(lines)


# Singleton
report_engine = ReportEngine()
