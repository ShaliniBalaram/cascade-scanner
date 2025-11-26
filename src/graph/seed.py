"""Database seeding from YAML configs."""

from pathlib import Path

import yaml
from loguru import logger

from src.graph.connection import get_connection
from src.graph.queries import (
    CREATE_ASSET, CREATE_FRAGILITY_CURVE, CREATE_CASCADE,
    CREATE_CONSTRAINTS, GET_STATS
)
from src.utils.config import get_project_root


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_schema():
    """Create constraints."""
    logger.info("Setting up schema...")
    conn = get_connection()
    for constraint in CREATE_CONSTRAINTS:
        try:
            conn.execute_write(constraint)
        except Exception as e:
            logger.debug(f"Constraint exists: {e}")


def seed_assets() -> int:
    """Load assets from YAML."""
    logger.info("Seeding assets...")
    path = get_project_root() / "config" / "assets" / "chennai_assets.yaml"
    if not path.exists():
        logger.error(f"Not found: {path}")
        return 0

    config = load_yaml(path)
    conn = get_connection()
    count = 0

    for asset in config.get("assets", []):
        params = {
            "asset_id": asset["asset_id"],
            "name": asset["name"],
            "asset_type": asset["asset_type"],
            "lat": asset["lat"],
            "lon": asset["lon"],
            "elevation_m": asset.get("elevation_m", 0),
            "attributes": asset.get("attributes", {}),
        }
        conn.execute_write(CREATE_ASSET, params)
        count += 1

    logger.info(f"Seeded {count} assets")
    return count


def seed_fragility_curves() -> int:
    """Load fragility curves from YAML."""
    logger.info("Seeding fragility curves...")
    path = get_project_root() / "config" / "fragility_curves" / "chennai_v1.yaml"
    if not path.exists():
        logger.error(f"Not found: {path}")
        return 0

    config = load_yaml(path)
    conn = get_connection()
    count = 0

    for curve in config.get("fragility_curves", []):
        trigger = curve.get("trigger", {})
        params = {
            "curve_id": curve["curve_id"],
            "asset_type": curve["asset_type"],
            "hazard_type": curve.get("hazard_type", "flood"),
            "trigger_depth_m": trigger.get("depth_m", 0),
            "trigger_duration_h": trigger.get("duration_hours", 0),
            "cascade_type": curve.get("cascade_type", ""),
            "probability": curve.get("probability", 0),
            "consequence_severity": curve.get("consequence_severity", "medium"),
            "recommended_action": curve.get("recommended_action", ""),
            "latency_hours": curve.get("latency_hours", 1),
            "source": curve.get("source", ""),
        }
        conn.execute_write(CREATE_FRAGILITY_CURVE, params)
        count += 1

    logger.info(f"Seeded {count} curves")
    return count


def seed_cascades() -> int:
    """Create cascade relationships."""
    logger.info("Seeding cascade relationships...")
    conn = get_connection()

    cascades = [
        ("substation_tondiarpet", "hospital_stanley", "power_dependency", 0.85, 0.5, "Power grid dependency"),
        ("substation_porur", "hospital_miot", "power_dependency", 0.80, 0.5, "Power grid dependency"),
        ("road_nh16_kathivakkam", "hospital_stanley", "access_dependency", 0.75, 1.0, "Primary access route"),
        ("road_anna_salai", "hospital_apollo", "access_dependency", 0.80, 0.5, "Main access road"),
        ("road_ecr_adyar", "hospital_fortis", "access_dependency", 0.85, 0.5, "ECR crossing critical"),
        ("substation_tondiarpet", "wwtp_kodungaiyur", "power_dependency", 0.90, 0.25, "STP pump power"),
        ("wwtp_kodungaiyur", "hospital_stanley", "contamination_risk", 0.60, 6.0, "Water contamination"),
        ("substation_porur", "wwtp_nesapakkam", "power_dependency", 0.85, 0.25, "STP power"),
    ]

    count = 0
    for src, tgt, ctype, prob, latency, desc in cascades:
        try:
            conn.execute_write(CREATE_CASCADE, {
                "source_id": src,
                "target_id": tgt,
                "cascade_type": ctype,
                "probability": prob,
                "latency_hours": latency,
                "description": desc,
            })
            count += 1
        except Exception as e:
            logger.warning(f"Cascade failed {src}->{tgt}: {e}")

    logger.info(f"Created {count} cascades")
    return count


def seed_database() -> dict:
    """Full database seeding."""
    logger.info("=" * 50)
    logger.info("SEEDING DATABASE")
    logger.info("=" * 50)

    setup_schema()
    assets = seed_assets()
    curves = seed_fragility_curves()
    cascades = seed_cascades()

    conn = get_connection()
    stats = conn.execute_query(GET_STATS)

    logger.info("=" * 50)
    logger.info(f"Done: {assets} assets, {curves} curves, {cascades} cascades")
    logger.info("=" * 50)

    return {"assets": assets, "curves": curves, "cascades": cascades, "stats": stats}


if __name__ == "__main__":
    seed_database()
