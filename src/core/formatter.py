"""Output formatters for stakeholders."""

import json
from src.graph.models import ScanResult


class EmergencyManagerFormatter:
    """Markdown action list for emergency managers."""

    def format(self, result: ScanResult) -> str:
        lines = [
            f"**ALERT: Flood-Cascade Event {'Active' if result.alerts else 'Monitor'}**",
            f"**Location:** {result.location.title()} | **Time:** {result.timestamp.strftime('%Y-%m-%d %H:%M')} IST",
            f"**Event ID:** {result.event_id}",
            "",
            "**CONDITIONS:**",
            f"- 24h Rain: {result.hazard_state.rainfall_24h_mm:.1f}mm",
            f"- Est. Depth: {result.hazard_state.depth_m:.2f}m",
            f"- Risk: {result.hazard_state.flood_risk_level.upper()}",
            "",
        ]

        if result.alerts:
            lines.append("**CASCADE RISKS:**")
            for i, a in enumerate(result.alerts[:10], 1):
                sev = a.consequence_severity.upper()
                lines.append(f"{i}. [{sev}] **{a.asset_name}** ({a.risk_score:.0f}%)")
                lines.append(f"   → {a.cascade_type} | ETA: {a.eta_failure_hours:.1f}h")
                lines.append(f"   **Action:** {a.recommended_action}")
                lines.append("")
        else:
            lines.append("**All systems normal.**")

        lines.extend([
            "---",
            f"**Summary:** {result.summary}",
            f"*Scan: {result.scan_duration_seconds:.2f}s*",
        ])

        return "\n".join(lines)


class ResearcherFormatter:
    """JSON with full details."""

    def format(self, result: ScanResult) -> dict:
        return {
            "event_id": result.event_id,
            "timestamp": result.timestamp.isoformat(),
            "location": result.location,
            "time_mode": result.time_mode,
            "scan_duration_seconds": result.scan_duration_seconds,
            "hazard_state": {
                "type": result.hazard_state.hazard_type,
                "source": result.hazard_state.source,
                "depth_m": result.hazard_state.depth_m,
                "duration_h": result.hazard_state.duration_h,
                "rainfall_24h_mm": result.hazard_state.rainfall_24h_mm,
                "flood_risk_level": result.hazard_state.flood_risk_level,
            },
            "alerts": [a.to_dict() for a in result.alerts],
            "safe_assets": result.safe_assets,
            "summary": result.summary,
        }

    def to_json(self, result: ScanResult) -> str:
        return json.dumps(self.format(result), indent=2, default=str)


def format_output(result: ScanResult, stakeholder: str = "emergency_manager") -> str:
    if stakeholder == "researcher":
        return ResearcherFormatter().to_json(result)
    return EmergencyManagerFormatter().format(result)
