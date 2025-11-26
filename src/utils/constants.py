"""Project-wide constants."""

HAZARD_TYPES = ["flood", "pluvial_flood", "fluvial_flood", "coastal_flood", "waterlogging"]

ASSET_TYPES = ["hospital", "substation", "wastewater_plant", "evacuation_route", "fire_station"]

CASCADE_TYPES = [
    "critical_service_disruption",
    "access_blocked",
    "power_outage",
    "water_contamination",
]

TIME_MODES = ["nowcast", "forecast", "diagnostic"]

STAKEHOLDER_TYPES = ["emergency_manager", "researcher", "public"]

RISK_LEVELS = {
    "critical": {"min": 0.85, "color": "#FF0000", "label": "CRITICAL"},
    "high": {"min": 0.70, "color": "#FF6600", "label": "HIGH"},
    "medium": {"min": 0.50, "color": "#FFCC00", "label": "MEDIUM"},
    "low": {"min": 0.25, "color": "#00CC00", "label": "LOW"},
    "minimal": {"min": 0.0, "color": "#00FF00", "label": "MINIMAL"},
}

CHENNAI_EPSG = 4326
CHENNAI_UTM_EPSG = 32644

SENTINEL1_PARAMS = {
    "collection": "COPERNICUS/S1_GRD",
    "polarization": "VV",
    "orbit": "DESCENDING",
    "resolution_m": 10,
}

FLOOD_THRESHOLDS = {
    "water_threshold_db": -15,
    "change_threshold_db": -3,
    "min_area_sqm": 1000,
}
