"""Temporal analysis module for flood risk trends."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
import numpy as np

@dataclass
class TemporalDataPoint:
    """Single point in time series."""
    timestamp: datetime
    rainfall_mm: float
    flood_depth_m: float
    risk_level: str
    alerts_count: int
    assets_at_risk: List[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Result of temporal trend analysis."""
    period_start: datetime
    period_end: datetime
    total_days: int

    # Rainfall stats
    total_rainfall_mm: float
    avg_daily_rainfall_mm: float
    max_daily_rainfall_mm: float
    rainy_days: int

    # Risk stats
    high_risk_days: int
    moderate_risk_days: int
    low_risk_days: int

    # Trend indicators
    rainfall_trend: str  # "increasing", "decreasing", "stable"
    risk_trend: str

    # Seasonality
    peak_risk_month: Optional[str] = None
    monsoon_intensity: str = "normal"  # "below_normal", "normal", "above_normal"

    # Alerts summary
    total_alerts: int = 0
    most_affected_assets: List[str] = field(default_factory=list)


class TemporalAnalyzer:
    """Analyze temporal patterns in flood risk data."""

    def __init__(self, data_dir: str = "data/temporal"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.data_dir / "scan_history.json"
        self._load_history()

    def _load_history(self):
        """Load historical scan data."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.history = [
                    TemporalDataPoint(
                        timestamp=datetime.fromisoformat(d['timestamp']),
                        rainfall_mm=d['rainfall_mm'],
                        flood_depth_m=d['flood_depth_m'],
                        risk_level=d['risk_level'],
                        alerts_count=d['alerts_count'],
                        assets_at_risk=d.get('assets_at_risk', [])
                    )
                    for d in data
                ]
        else:
            self.history = []

    def _save_history(self):
        """Save historical scan data."""
        data = [
            {
                'timestamp': d.timestamp.isoformat(),
                'rainfall_mm': d.rainfall_mm,
                'flood_depth_m': d.flood_depth_m,
                'risk_level': d.risk_level,
                'alerts_count': d.alerts_count,
                'assets_at_risk': d.assets_at_risk
            }
            for d in self.history
        ]
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_scan(self, scan_result) -> None:
        """Record a scan result for temporal analysis."""
        data_point = TemporalDataPoint(
            timestamp=scan_result.timestamp if hasattr(scan_result, 'timestamp') else datetime.now(),
            rainfall_mm=scan_result.hazard_state.rainfall_24h_mm if hasattr(scan_result, 'hazard_state') else 0,
            flood_depth_m=scan_result.hazard_state.depth_m if hasattr(scan_result, 'hazard_state') else 0,
            risk_level=scan_result.hazard_state.flood_risk_level if hasattr(scan_result, 'hazard_state') else 'minimal',
            alerts_count=len(scan_result.alerts) if hasattr(scan_result, 'alerts') else 0,
            assets_at_risk=[a.asset_id for a in scan_result.alerts[:10]] if hasattr(scan_result, 'alerts') else []
        )
        self.history.append(data_point)
        self._save_history()

    def analyze_period(
        self,
        days_back: int = 30,
        end_date: Optional[datetime] = None
    ) -> TrendAnalysis:
        """Analyze trends over a specified period."""
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Filter data for period
        period_data = [
            d for d in self.history
            if start_date <= d.timestamp <= end_date
        ]

        if not period_data:
            # Return empty analysis with simulated historical data
            return self._generate_simulated_analysis(start_date, end_date, days_back)

        # Calculate stats
        rainfall_values = [d.rainfall_mm for d in period_data]
        risk_levels = [d.risk_level for d in period_data]

        total_rainfall = sum(rainfall_values)
        avg_rainfall = np.mean(rainfall_values) if rainfall_values else 0
        max_rainfall = max(rainfall_values) if rainfall_values else 0
        rainy_days = sum(1 for r in rainfall_values if r > 0.1)

        high_risk = sum(1 for r in risk_levels if r in ['high', 'severe'])
        moderate_risk = sum(1 for r in risk_levels if r == 'moderate')
        low_risk = sum(1 for r in risk_levels if r in ['low', 'minimal'])

        # Calculate trends
        if len(rainfall_values) >= 7:
            first_half = np.mean(rainfall_values[:len(rainfall_values)//2])
            second_half = np.mean(rainfall_values[len(rainfall_values)//2:])
            if second_half > first_half * 1.2:
                rainfall_trend = "increasing"
            elif second_half < first_half * 0.8:
                rainfall_trend = "decreasing"
            else:
                rainfall_trend = "stable"
        else:
            rainfall_trend = "insufficient_data"

        # Risk trend
        recent_risks = risk_levels[-7:] if len(risk_levels) >= 7 else risk_levels
        high_count = sum(1 for r in recent_risks if r in ['high', 'severe'])
        if high_count >= 3:
            risk_trend = "elevated"
        elif high_count >= 1:
            risk_trend = "moderate"
        else:
            risk_trend = "stable"

        # Most affected assets
        all_assets = []
        for d in period_data:
            all_assets.extend(d.assets_at_risk)
        asset_counts = {}
        for a in all_assets:
            asset_counts[a] = asset_counts.get(a, 0) + 1
        most_affected = sorted(asset_counts.keys(), key=lambda x: asset_counts[x], reverse=True)[:5]

        return TrendAnalysis(
            period_start=start_date,
            period_end=end_date,
            total_days=days_back,
            total_rainfall_mm=total_rainfall,
            avg_daily_rainfall_mm=avg_rainfall,
            max_daily_rainfall_mm=max_rainfall,
            rainy_days=rainy_days,
            high_risk_days=high_risk,
            moderate_risk_days=moderate_risk,
            low_risk_days=low_risk,
            rainfall_trend=rainfall_trend,
            risk_trend=risk_trend,
            total_alerts=sum(d.alerts_count for d in period_data),
            most_affected_assets=most_affected
        )

    def _generate_simulated_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        days: int
    ) -> TrendAnalysis:
        """Generate simulated historical analysis for demo purposes."""
        # Simulate Chennai monsoon patterns
        month = end_date.month

        # Northeast monsoon (Oct-Dec) is peak flood season for Chennai
        if month in [10, 11, 12]:
            total_rainfall = np.random.uniform(200, 450)
            high_risk_days = np.random.randint(3, 10)
            monsoon = "above_normal" if total_rainfall > 350 else "normal"
            peak_month = "November"
        elif month in [6, 7, 8, 9]:
            # Southwest monsoon - moderate
            total_rainfall = np.random.uniform(50, 150)
            high_risk_days = np.random.randint(0, 3)
            monsoon = "normal"
            peak_month = "August"
        else:
            # Dry season
            total_rainfall = np.random.uniform(5, 50)
            high_risk_days = 0
            monsoon = "below_normal"
            peak_month = None

        rainy_days = int(total_rainfall / 15)

        return TrendAnalysis(
            period_start=start_date,
            period_end=end_date,
            total_days=days,
            total_rainfall_mm=total_rainfall,
            avg_daily_rainfall_mm=total_rainfall / days,
            max_daily_rainfall_mm=total_rainfall * 0.15,
            rainy_days=rainy_days,
            high_risk_days=high_risk_days,
            moderate_risk_days=max(0, rainy_days - high_risk_days),
            low_risk_days=days - rainy_days,
            rainfall_trend="stable",
            risk_trend="elevated" if high_risk_days > 5 else "stable",
            peak_risk_month=peak_month,
            monsoon_intensity=monsoon,
            total_alerts=high_risk_days * 3,
            most_affected_assets=["substation_tondiarpet", "wwtp_kodungaiyur", "road_nh16"]
        )

    def get_historical_series(
        self,
        days_back: int = 30,
        metric: str = "rainfall"
    ) -> List[Dict[str, Any]]:
        """Get time series data for visualization."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Filter and prepare data
        period_data = [
            d for d in self.history
            if start_date <= d.timestamp <= end_date
        ]

        if period_data:
            return [
                {
                    "date": d.timestamp.strftime("%Y-%m-%d"),
                    "value": getattr(d, f"{metric}_mm" if metric == "rainfall" else metric, 0),
                    "risk_level": d.risk_level
                }
                for d in sorted(period_data, key=lambda x: x.timestamp)
            ]

        # Generate simulated time series for demo
        return self._generate_simulated_series(start_date, end_date, metric)

    def _generate_simulated_series(
        self,
        start_date: datetime,
        end_date: datetime,
        metric: str
    ) -> List[Dict[str, Any]]:
        """Generate simulated time series for demo."""
        series = []
        current = start_date

        while current <= end_date:
            # Simulate rainfall patterns
            month = current.month
            day_of_week = current.weekday()

            # Base rainfall varies by season
            if month in [10, 11, 12]:
                base = 8.0  # Northeast monsoon
            elif month in [6, 7, 8, 9]:
                base = 3.0  # Southwest monsoon
            else:
                base = 0.5  # Dry season

            # Add randomness
            rainfall = max(0, base + np.random.normal(0, base * 0.5))

            # Risk level based on rainfall
            if rainfall > 50:
                risk = "high"
            elif rainfall > 20:
                risk = "moderate"
            elif rainfall > 5:
                risk = "low"
            else:
                risk = "minimal"

            series.append({
                "date": current.strftime("%Y-%m-%d"),
                "value": round(rainfall, 1),
                "risk_level": risk
            })

            current += timedelta(days=1)

        return series

    def compare_periods(
        self,
        period1_start: datetime,
        period1_end: datetime,
        period2_start: datetime,
        period2_end: datetime
    ) -> Dict[str, Any]:
        """Compare two time periods."""
        days1 = (period1_end - period1_start).days
        days2 = (period2_end - period2_start).days

        analysis1 = self.analyze_period(days1, period1_end)
        analysis2 = self.analyze_period(days2, period2_end)

        return {
            "period1": {
                "start": period1_start.isoformat(),
                "end": period1_end.isoformat(),
                "total_rainfall_mm": analysis1.total_rainfall_mm,
                "high_risk_days": analysis1.high_risk_days,
                "total_alerts": analysis1.total_alerts
            },
            "period2": {
                "start": period2_start.isoformat(),
                "end": period2_end.isoformat(),
                "total_rainfall_mm": analysis2.total_rainfall_mm,
                "high_risk_days": analysis2.high_risk_days,
                "total_alerts": analysis2.total_alerts
            },
            "comparison": {
                "rainfall_change_pct": ((analysis2.total_rainfall_mm - analysis1.total_rainfall_mm) / max(analysis1.total_rainfall_mm, 1)) * 100,
                "risk_change": analysis2.high_risk_days - analysis1.high_risk_days,
                "alerts_change": analysis2.total_alerts - analysis1.total_alerts
            }
        }


# Singleton instance
temporal_analyzer = TemporalAnalyzer()
