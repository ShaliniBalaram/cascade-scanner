"""3-layer cascade scan engine."""

import uuid
from datetime import datetime, timedelta, timezone

from loguru import logger

from src.data_sources.imd_client import imd_client
from src.data_sources.gee_client import gee_client
from src.graph.connection import get_connection
from src.graph.models import Alert, HazardState, ScanResult
from src.graph.queries import GET_ALL_ASSETS, GET_CURVES_FOR_ASSET_TYPE, GET_DOWNSTREAM_CASCADES
from src.utils.config import settings


class CascadeScanner:
    """3-layer probabilistic scan engine."""

    def __init__(self):
        self.conn = get_connection()
        self.threshold = settings.scanner.alert_threshold_probability

    def execute_scan(
        self,
        location: str = "chennai",
        time_mode: str = "nowcast",
        stakeholder: str = "emergency_manager",
    ) -> ScanResult:
        """Execute full 3-layer scan."""
        start = datetime.now(timezone.utc)
        event_id = f"CH-{start.strftime('%Y%m%d-%H%M')}"

        logger.info(f"Scan {event_id}: {location}, {time_mode}")

        # Layer 1: Hazard Detection
        logger.info("L1: Detecting hazards...")
        hazard = self._detect_hazards(time_mode)

        # Layer 2: Exposure Mapping
        logger.info("L2: Mapping exposure...")
        exposed = self._map_exposure(hazard)

        # Layer 3: Fragility Evaluation
        logger.info("L3: Evaluating fragility...")
        alerts, safe = self._evaluate_fragility(exposed, hazard)

        duration = (datetime.now(timezone.utc) - start).total_seconds()
        summary = self._summary(alerts, safe, hazard)

        logger.info(f"Scan done: {len(alerts)} alerts, {duration:.2f}s")

        return ScanResult(
            event_id=event_id,
            timestamp=start,
            location=location,
            time_mode=time_mode,
            hazard_state=hazard,
            alerts=alerts,
            safe_assets=safe,
            summary=summary,
            scan_duration_seconds=duration,
            metadata={"stakeholder": stakeholder},
        )

    def _detect_hazards(self, time_mode: str) -> HazardState:
        """Layer 1: Get current hazard conditions."""
        now = datetime.now(timezone.utc)

        # Get rainfall
        if time_mode == "nowcast":
            rain = imd_client.get_current_rainfall()
            forecast = imd_client.get_rainfall_forecast(days=3)
        else:
            rain = {"last_24h_precipitation_mm": 0}
            forecast = imd_client.get_rainfall_forecast(days=7)

        rainfall_mm = rain.get("last_24h_precipitation_mm", 0) or 0
        risk = forecast.get("flood_risk", {})

        # Estimate depth from rainfall
        depth = self._estimate_depth(rainfall_mm)

        # Try SAR detection if significant rain
        polygons = []
        if rainfall_mm > 20 and time_mode == "nowcast":
            try:
                sar = gee_client.get_latest_flood_extent()
                polygons = sar.get("flood_polygons", {}).get("features", [])
                if sar.get("flood_area_sqkm", 0) > 0:
                    depth = max(depth, 0.3)
            except Exception as e:
                logger.warning(f"SAR skipped: {e}")

        return HazardState(
            hazard_type="flood",
            source="sentinel1_sar" if polygons else "imd_precipitation",
            timestamp=now,
            depth_m=depth,
            duration_h=self._estimate_duration(rainfall_mm),
            rainfall_24h_mm=rainfall_mm,
            flood_risk_level=risk.get("level", "minimal"),
            flood_polygons=polygons,
            bbox={
                "north": settings.chennai.bbox.north,
                "south": settings.chennai.bbox.south,
                "east": settings.chennai.bbox.east,
                "west": settings.chennai.bbox.west,
            },
        )

    def _estimate_depth(self, mm: float) -> float:
        """Estimate flood depth from rainfall."""
        if mm < 30: return 0.0
        if mm < 50: return 0.1
        if mm < 80: return 0.25
        if mm < 120: return 0.4
        if mm < 180: return 0.6
        return 0.8 + (mm - 180) / 200

    def _estimate_duration(self, mm: float) -> float:
        """Estimate flood duration in hours."""
        if mm < 30: return 0
        if mm < 80: return 2
        if mm < 150: return 6
        return 12

    def _map_exposure(self, hazard: HazardState) -> list:
        """Layer 2: Find exposed assets."""
        if hazard.depth_m == 0:
            return []

        assets = self.conn.execute_query(GET_ALL_ASSETS)
        exposed = []

        for rec in assets:
            a = rec.get("a", {})
            elev = a.get("elevation_m", 0)
            # Low-lying assets exposed
            if elev <= 10 and hazard.depth_m > 0.1:
                exposed.append(a)
            elif elev <= 5:
                exposed.append(a)

        logger.info(f"Exposed: {len(exposed)}/{len(assets)} assets")
        return exposed

    def _evaluate_fragility(self, exposed: list, hazard: HazardState) -> tuple:
        """Layer 3: Check threshold exceedances."""
        alerts = []
        safe = []

        for asset in exposed:
            asset_type = asset.get("asset_type")
            asset_id = asset.get("asset_id")
            asset_name = asset.get("name", asset_id)

            curves = self.conn.execute_query(
                GET_CURVES_FOR_ASSET_TYPE, {"asset_type": asset_type}
            )

            exceeded = False
            for rec in curves:
                c = rec.get("f", {})
                depth_th = c.get("trigger_depth_m", 0)
                dur_th = c.get("trigger_duration_h", 0)
                prob = c.get("probability", 0)

                if hazard.depth_m >= depth_th and (hazard.duration_h >= dur_th or dur_th == 0):
                    exceeded = True
                    if prob >= self.threshold:
                        risk = self._risk_score(prob, c.get("consequence_severity", "medium"))
                        alert = Alert(
                            alert_id=f"{asset_id}-{uuid.uuid4().hex[:6]}",
                            asset_id=asset_id,
                            asset_name=asset_name,
                            hazard_type=hazard.hazard_type,
                            cascade_type=c.get("cascade_type", ""),
                            risk_score=risk,
                            probability=prob,
                            consequence_severity=c.get("consequence_severity", "medium"),
                            recommended_action=c.get("recommended_action", ""),
                            eta_failure_hours=c.get("latency_hours", 1),
                        )
                        alerts.append(alert)
                        self._check_cascades(alert, alerts, hazard)

            if not exceeded:
                safe.append({"asset_id": asset_id, "asset_name": asset_name})

        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return alerts, safe

    def _risk_score(self, prob: float, severity: str) -> float:
        """Calculate risk score."""
        weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        return round(prob * weights.get(severity, 0.5) * 100, 1)

    def _check_cascades(self, trigger: Alert, alerts: list, hazard: HazardState):
        """Add downstream cascade alerts."""
        try:
            cascades = self.conn.execute_query(
                GET_DOWNSTREAM_CASCADES, {"asset_id": trigger.asset_id}
            )
            for rec in cascades:
                ds = rec.get("downstream", {})
                rels = rec.get("r", [{}])
                rel = rels[0] if isinstance(rels, list) else rels

                combined_prob = trigger.probability * rel.get("probability", 0)
                if combined_prob >= self.threshold:
                    alerts.append(Alert(
                        alert_id=f"{ds.get('asset_id')}-cascade-{uuid.uuid4().hex[:4]}",
                        asset_id=ds.get("asset_id"),
                        asset_name=ds.get("name"),
                        hazard_type="cascade",
                        cascade_type=rel.get("cascade_type", ""),
                        risk_score=trigger.risk_score * rel.get("probability", 0),
                        probability=combined_prob,
                        consequence_severity="high",
                        recommended_action=f"Cascade from {trigger.asset_name}",
                        eta_failure_hours=trigger.eta_failure_hours + rel.get("latency_hours", 1),
                    ))
        except Exception as e:
            logger.warning(f"Cascade check failed: {e}")

    def _summary(self, alerts: list, safe: list, hazard: HazardState) -> str:
        if not alerts:
            return "No cascade risks detected." if hazard.depth_m == 0 else f"{len(safe)} assets below thresholds."

        crit = sum(1 for a in alerts if a.consequence_severity == "critical")
        high = sum(1 for a in alerts if a.consequence_severity == "high")
        parts = []
        if crit: parts.append(f"{crit} CRITICAL")
        if high: parts.append(f"{high} HIGH")
        return f"{len(alerts)} risks ({', '.join(parts)}). Action required."


scanner = CascadeScanner()
