"""Data models for knowledge graph."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Asset:
    """Critical infrastructure asset."""
    asset_id: str
    name: str
    asset_type: str
    lat: float
    lon: float
    elevation_m: float = 0.0
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "asset_type": self.asset_type,
            "lat": self.lat,
            "lon": self.lon,
            "elevation_m": self.elevation_m,
            **self.attributes,
        }


@dataclass
class FragilityCurve:
    """Threshold-based failure probability."""
    curve_id: str
    asset_type: str
    hazard_type: str
    trigger_depth_m: float
    trigger_duration_h: float = 0.0
    cascade_type: str = ""
    probability: float = 0.0
    consequence_severity: str = "medium"
    recommended_action: str = ""
    latency_hours: float = 1.0
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "curve_id": self.curve_id,
            "asset_type": self.asset_type,
            "hazard_type": self.hazard_type,
            "trigger_depth_m": self.trigger_depth_m,
            "trigger_duration_h": self.trigger_duration_h,
            "cascade_type": self.cascade_type,
            "probability": self.probability,
            "consequence_severity": self.consequence_severity,
            "recommended_action": self.recommended_action,
            "latency_hours": self.latency_hours,
            "source": self.source,
        }

    def exceeds_threshold(self, depth_m: float, duration_h: float = 0) -> bool:
        depth_ok = depth_m >= self.trigger_depth_m
        duration_ok = duration_h >= self.trigger_duration_h if self.trigger_duration_h > 0 else True
        return depth_ok and duration_ok


@dataclass
class Alert:
    """Generated cascade risk alert."""
    alert_id: str
    asset_id: str
    asset_name: str
    hazard_type: str
    cascade_type: str
    risk_score: float
    probability: float
    consequence_severity: str
    recommended_action: str
    eta_failure_hours: float
    validation_signals: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "hazard_type": self.hazard_type,
            "cascade_type": self.cascade_type,
            "risk_score": self.risk_score,
            "probability": self.probability,
            "consequence_severity": self.consequence_severity,
            "recommended_action": self.recommended_action,
            "eta_failure_hours": self.eta_failure_hours,
            "validation_signals": self.validation_signals,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HazardState:
    """Current hazard conditions."""
    hazard_type: str
    source: str
    timestamp: datetime
    depth_m: float = 0.0
    duration_h: float = 0.0
    area_sqkm: float = 0.0
    rainfall_24h_mm: float = 0.0
    flood_risk_level: str = "minimal"
    flood_polygons: list = field(default_factory=list)
    bbox: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    """Complete scan output."""
    event_id: str
    timestamp: datetime
    location: str
    time_mode: str
    hazard_state: HazardState
    alerts: list
    safe_assets: list
    summary: str
    scan_duration_seconds: float
    metadata: dict = field(default_factory=dict)
