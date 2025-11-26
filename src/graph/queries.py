"""Cypher query templates."""

# Schema
CREATE_CONSTRAINTS = [
    "CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE",
    "CREATE CONSTRAINT curve_id IF NOT EXISTS FOR (f:FragilityCurve) REQUIRE f.curve_id IS UNIQUE",
]

# Assets
CREATE_ASSET = """
MERGE (a:Asset {asset_id: $asset_id})
SET a.name = $name,
    a.asset_type = $asset_type,
    a.lat = $lat,
    a.lon = $lon,
    a.elevation_m = $elevation_m,
    a += $attributes
RETURN a
"""

GET_ALL_ASSETS = """
MATCH (a:Asset)
RETURN a
ORDER BY a.asset_type, a.name
"""

GET_ASSETS_BY_TYPE = """
MATCH (a:Asset {asset_type: $asset_type})
RETURN a
"""

# Fragility Curves
CREATE_FRAGILITY_CURVE = """
MERGE (f:FragilityCurve {curve_id: $curve_id})
SET f.asset_type = $asset_type,
    f.hazard_type = $hazard_type,
    f.trigger_depth_m = $trigger_depth_m,
    f.trigger_duration_h = $trigger_duration_h,
    f.cascade_type = $cascade_type,
    f.probability = $probability,
    f.consequence_severity = $consequence_severity,
    f.recommended_action = $recommended_action,
    f.latency_hours = $latency_hours,
    f.source = $source
RETURN f
"""

GET_CURVES_FOR_ASSET_TYPE = """
MATCH (f:FragilityCurve {asset_type: $asset_type})
RETURN f
"""

GET_ALL_CURVES = """
MATCH (f:FragilityCurve)
RETURN f
ORDER BY f.asset_type, f.probability DESC
"""

# Cascade Relationships
CREATE_CASCADE = """
MATCH (src:Asset {asset_id: $source_id})
MATCH (tgt:Asset {asset_id: $target_id})
MERGE (src)-[r:CASCADE_TO]->(tgt)
SET r.cascade_type = $cascade_type,
    r.probability = $probability,
    r.latency_hours = $latency_hours,
    r.description = $description
RETURN src, r, tgt
"""

GET_DOWNSTREAM_CASCADES = """
MATCH (a:Asset {asset_id: $asset_id})-[r:CASCADE_TO*1..3]->(downstream:Asset)
RETURN downstream, r
"""

# Scan Queries
SCAN_THRESHOLD_EXCEEDANCES = """
MATCH (a:Asset)
MATCH (f:FragilityCurve {asset_type: a.asset_type, hazard_type: $hazard_type})
WHERE $depth >= f.trigger_depth_m
  AND ($duration >= f.trigger_duration_h OR f.trigger_duration_h = 0)
RETURN a, f,
       f.probability AS probability,
       f.cascade_type AS cascade_type,
       f.recommended_action AS action,
       f.latency_hours AS eta_hours,
       f.consequence_severity AS severity
ORDER BY f.probability DESC
"""

# Stats
GET_STATS = """
MATCH (a:Asset) WITH count(a) AS assets
MATCH (f:FragilityCurve) WITH assets, count(f) AS curves
MATCH ()-[r:CASCADE_TO]->()
RETURN assets, curves, count(r) AS cascades
"""

DELETE_ALL = "MATCH (n) DETACH DELETE n"
