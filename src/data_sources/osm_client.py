"""OpenStreetMap client for critical infrastructure extraction."""

from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd
from loguru import logger
from shapely.geometry import Point

from src.utils.config import settings


class OSMClient:
    """Client for OSM Overpass API to extract critical assets."""

    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        self.timeout = 120
        self.bbox = settings.chennai.bbox
        self.cache_dir = Path("data/cache/osm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _bbox_str(self) -> str:
        """Overpass bbox format: south,west,north,east."""
        return f"{self.bbox.south},{self.bbox.west},{self.bbox.north},{self.bbox.east}"

    def _query(self, query: str) -> dict:
        """Execute Overpass query."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.overpass_url, data={"data": query})
            resp.raise_for_status()
            return resp.json()

    def get_hospitals(self) -> gpd.GeoDataFrame:
        """Extract hospitals."""
        logger.info("Fetching hospitals...")
        query = f"""
        [out:json][timeout:60];
        (node["amenity"="hospital"]({self._bbox_str()});
         way["amenity"="hospital"]({self._bbox_str()}););
        out center;
        """
        data = self._query(query)

        rows = []
        for el in data.get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                tags = el.get("tags", {})
                rows.append({
                    "osm_id": el["id"],
                    "name": tags.get("name", f"Hospital_{el['id']}"),
                    "lat": lat, "lon": lon,
                    "asset_type": "hospital",
                    "geometry": Point(lon, lat),
                })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame()
        logger.info(f"Found {len(gdf)} hospitals")
        return gdf

    def get_substations(self) -> gpd.GeoDataFrame:
        """Extract electrical substations."""
        logger.info("Fetching substations...")
        query = f"""
        [out:json][timeout:60];
        (node["power"="substation"]({self._bbox_str()});
         way["power"="substation"]({self._bbox_str()}););
        out center;
        """
        data = self._query(query)

        rows = []
        for el in data.get("elements", []):
            lat = el.get("lat") or el.get("center", {}).get("lat")
            lon = el.get("lon") or el.get("center", {}).get("lon")
            if lat and lon:
                tags = el.get("tags", {})
                rows.append({
                    "osm_id": el["id"],
                    "name": tags.get("name", f"Substation_{el['id']}"),
                    "lat": lat, "lon": lon,
                    "asset_type": "substation",
                    "voltage": tags.get("voltage"),
                    "geometry": Point(lon, lat),
                })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame()
        logger.info(f"Found {len(gdf)} substations")
        return gdf

    def get_critical_roads(self) -> gpd.GeoDataFrame:
        """Extract major roads (NH, SH)."""
        logger.info("Fetching critical roads...")
        query = f"""
        [out:json][timeout:90];
        (way["highway"="trunk"]({self._bbox_str()});
         way["highway"="primary"]({self._bbox_str()});
         way["ref"~"NH"]({self._bbox_str()}););
        out geom;
        """
        data = self._query(query)

        rows = []
        seen_refs = set()
        for el in data.get("elements", []):
            if el.get("type") == "way" and el.get("geometry"):
                tags = el.get("tags", {})
                ref = tags.get("ref", "")

                # Skip duplicates
                if ref and ref in seen_refs:
                    continue
                if ref:
                    seen_refs.add(ref)

                coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
                if coords:
                    center_lon = sum(c[0] for c in coords) / len(coords)
                    center_lat = sum(c[1] for c in coords) / len(coords)
                    rows.append({
                        "osm_id": el["id"],
                        "name": tags.get("name", ref or f"Road_{el['id']}"),
                        "ref": ref,
                        "lat": center_lat, "lon": center_lon,
                        "asset_type": "evacuation_route",
                        "geometry": Point(center_lon, center_lat),
                    })

        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame()
        logger.info(f"Found {len(gdf)} critical roads")
        return gdf

    def get_all_assets(self, save_cache: bool = True) -> gpd.GeoDataFrame:
        """Fetch all critical assets."""
        logger.info("Fetching all Chennai critical assets...")

        hospitals = self.get_hospitals()
        substations = self.get_substations()
        roads = self.get_critical_roads()

        all_assets = pd.concat([hospitals, substations, roads], ignore_index=True)
        gdf = gpd.GeoDataFrame(all_assets, crs="EPSG:4326")
        gdf["asset_id"] = gdf.apply(lambda r: f"{r['asset_type']}_{r['osm_id']}", axis=1)

        if save_cache:
            cache_path = self.cache_dir / "chennai_osm_assets.geojson"
            gdf.to_file(cache_path, driver="GeoJSON")
            logger.info(f"Cached {len(gdf)} assets")

        return gdf

    def load_cached(self) -> gpd.GeoDataFrame:
        """Load cached assets."""
        path = self.cache_dir / "chennai_osm_assets.geojson"
        if path.exists():
            return gpd.read_file(path)
        return gpd.GeoDataFrame()


osm_client = OSMClient()
