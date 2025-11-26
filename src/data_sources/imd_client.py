"""IMD precipitation data client using Open-Meteo API."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from src.utils.config import settings


class IMDClient:
    """Client for precipitation data via Open-Meteo (free, no API key)."""

    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.archive_url = "https://archive-api.open-meteo.com/v1/archive"
        self.timeout = settings.imd.timeout_seconds
        self.cache_dir = Path("data/cache/imd")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lat = settings.chennai.center["lat"]
        self.lon = settings.chennai.center["lon"]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str, ttl_hours: int = 1) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.utcnow() - cached_at < timedelta(hours=ttl_hours):
                return data["data"]
        except Exception:
            pass
        return None

    def _write_cache(self, key: str, data: dict):
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump({"cached_at": datetime.utcnow().isoformat(), "data": data}, f)

    def get_current_rainfall(self, lat: float = None, lon: float = None) -> dict:
        """Get current/recent rainfall."""
        lat = lat or self.lat
        lon = lon or self.lon
        cache_key = f"current_{lat}_{lon}"

        cached = self._read_cache(cache_key)
        if cached:
            return cached

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "precipitation,rain",
                "daily": "precipitation_sum",
                "timezone": "Asia/Kolkata",
                "forecast_days": 2,
            }

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(self.base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            precip = hourly.get("precipitation", [])

            # Sum last 24 hours
            now = datetime.now()
            last_24h = 0.0
            for i, t in enumerate(times):
                t_dt = datetime.fromisoformat(t)
                if now - timedelta(hours=24) <= t_dt <= now:
                    last_24h += precip[i] if i < len(precip) else 0

            result = {
                "location": {"lat": lat, "lon": lon},
                "timestamp": datetime.utcnow().isoformat(),
                "last_24h_precipitation_mm": round(last_24h, 1),
                "rainfall_category": self._categorize(last_24h),
                "source": "Open-Meteo API",
            }

            self._write_cache(cache_key, result)
            logger.info(f"Current rainfall: {last_24h:.1f}mm")
            return result

        except Exception as e:
            logger.error(f"Rainfall fetch failed: {e}")
            return {"error": str(e), "last_24h_precipitation_mm": 0}

    def get_rainfall_forecast(self, lat: float = None, lon: float = None, days: int = 7) -> dict:
        """Get rainfall forecast."""
        lat = lat or self.lat
        lon = lon or self.lon
        cache_key = f"forecast_{lat}_{lon}_{days}"

        cached = self._read_cache(cache_key)
        if cached:
            return cached

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,rain_sum",
                "timezone": "Asia/Kolkata",
                "forecast_days": min(days, 7),
            }

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(self.base_url, params=params)
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])

            forecast = [
                {"date": times[i], "precipitation_mm": precip[i], "category": self._categorize(precip[i])}
                for i in range(len(times))
            ]

            max_precip = max(precip) if precip else 0

            result = {
                "location": {"lat": lat, "lon": lon},
                "generated_at": datetime.utcnow().isoformat(),
                "daily_forecast": forecast,
                "max_expected_mm": round(max_precip, 1),
                "flood_risk": self._assess_risk(precip),
                "source": "Open-Meteo API",
            }

            self._write_cache(cache_key, result)
            logger.info(f"Forecast max: {max_precip:.1f}mm")
            return result

        except Exception as e:
            logger.error(f"Forecast fetch failed: {e}")
            return {"error": str(e)}

    def get_historical_rainfall(self, start: datetime, end: datetime, lat: float = None, lon: float = None) -> dict:
        """Get historical rainfall data."""
        lat = lat or self.lat
        lon = lon or self.lon

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "daily": "precipitation_sum",
                "timezone": "Asia/Kolkata",
            }

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(self.archive_url, params=params)
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])

            total = sum(p for p in precip if p)
            max_daily = max(precip) if precip else 0
            rainy_days = sum(1 for p in precip if p and p > 2.5)

            return {
                "period": {"start": start.isoformat(), "end": end.isoformat()},
                "statistics": {
                    "total_mm": round(total, 1),
                    "max_daily_mm": round(max_daily, 1),
                    "rainy_days": rainy_days,
                },
                "source": "Open-Meteo Archive",
            }

        except Exception as e:
            logger.error(f"Historical fetch failed: {e}")
            return {"error": str(e)}

    def _categorize(self, mm: float) -> str:
        """Categorize rainfall per IMD standards."""
        if mm < 2.5:
            return "no_rain"
        elif mm < 7.5:
            return "light"
        elif mm < 35.5:
            return "moderate"
        elif mm < 64.4:
            return "heavy"
        elif mm < 124.4:
            return "very_heavy"
        return "extremely_heavy"

    def _assess_risk(self, daily_precip: list) -> dict:
        """Assess flood risk from precipitation forecast."""
        if not daily_precip:
            return {"level": "unknown", "score": 0}

        max_daily = max(daily_precip)
        total_3day = sum(daily_precip[:3])

        score = min(100, (max_daily / 3) + (total_3day / 5))

        if score >= 70:
            level = "high"
        elif score >= 40:
            level = "moderate"
        elif score >= 20:
            level = "low"
        else:
            level = "minimal"

        return {"level": level, "score": round(score, 1)}


imd_client = IMDClient()
