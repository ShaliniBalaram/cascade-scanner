"""Google Earth Engine client for Sentinel-1 SAR flood detection."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import ee
from loguru import logger

from src.utils.config import settings
from src.utils.constants import FLOOD_THRESHOLDS, SENTINEL1_PARAMS


class GEEClient:
    """Client for GEE Sentinel-1 flood detection."""

    def __init__(self):
        self.initialized = False
        self.project_id = settings.gee.project_id
        self.key_path = settings.gee.service_account_key
        self.bbox = settings.chennai.bbox

    def authenticate(self) -> bool:
        """Authenticate with GEE using service account."""
        if self.initialized:
            return True

        try:
            if self.key_path and Path(self.key_path).exists():
                with open(self.key_path) as f:
                    sa = json.load(f)
                credentials = ee.ServiceAccountCredentials(sa['client_email'], self.key_path)
                ee.Initialize(credentials=credentials, project=self.project_id)
                logger.info("GEE authenticated via service account")
            else:
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
                logger.info("GEE authenticated interactively")

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"GEE auth failed: {e}")
            return False

    def get_chennai_geometry(self) -> ee.Geometry:
        """Get Chennai bounding box as EE geometry."""
        return ee.Geometry.Rectangle([
            self.bbox.west, self.bbox.south,
            self.bbox.east, self.bbox.north
        ])

    def get_sentinel1_collection(
        self,
        start_date: datetime,
        end_date: datetime,
        geometry: Optional[ee.Geometry] = None,
    ) -> ee.ImageCollection:
        """Get Sentinel-1 GRD collection for date range."""
        if not self.initialized:
            self.authenticate()

        geometry = geometry or self.get_chennai_geometry()

        collection = (
            ee.ImageCollection(SENTINEL1_PARAMS["collection"])
            .filterBounds(geometry)
            .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", SENTINEL1_PARAMS["polarization"]))
            .filter(ee.Filter.eq("orbitProperties_pass", SENTINEL1_PARAMS["orbit"]))
            .select(SENTINEL1_PARAMS["polarization"])
        )

        count = collection.size().getInfo()
        logger.info(f"Found {count} Sentinel-1 images")
        return collection

    def detect_flood_extent(
        self,
        event_date: datetime,
        baseline_days: int = 30,
        geometry: Optional[ee.Geometry] = None,
    ) -> dict:
        """
        Detect flood extent using SAR change detection.

        Compares event imagery to baseline to identify flooded areas.
        """
        if not self.initialized:
            self.authenticate()

        geometry = geometry or self.get_chennai_geometry()

        # Date ranges
        event_start = event_date - timedelta(days=3)
        event_end = event_date + timedelta(days=3)
        baseline_start = event_date - timedelta(days=baseline_days + 30)
        baseline_end = event_date - timedelta(days=baseline_days)

        logger.info(f"Flood detection for {event_date.strftime('%Y-%m-%d')}")

        # Baseline composite
        baseline = self.get_sentinel1_collection(baseline_start, baseline_end, geometry)
        baseline_composite = baseline.median().clip(geometry)

        # Event composite
        event = self.get_sentinel1_collection(event_start, event_end, geometry)
        event_composite = event.median().clip(geometry)

        # Change detection
        difference = event_composite.subtract(baseline_composite)
        flood_mask = difference.lt(FLOOD_THRESHOLDS["change_threshold_db"])
        water_mask = event_composite.lt(FLOOD_THRESHOLDS["water_threshold_db"])
        combined = flood_mask.Or(water_mask)

        # Calculate area
        flood_area = combined.multiply(ee.Image.pixelArea())
        stats = flood_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=SENTINEL1_PARAMS["resolution_m"],
            maxPixels=1e9,
        )
        area_sqkm = stats.getInfo().get(SENTINEL1_PARAMS["polarization"], 0) / 1e6

        # Vectorize flood polygons
        try:
            vectors = combined.selfMask().reduceToVectors(
                geometry=geometry,
                scale=SENTINEL1_PARAMS["resolution_m"],
                geometryType="polygon",
                eightConnected=True,
                maxPixels=1e8,
            )
            flood_geojson = vectors.filter(
                ee.Filter.gt("count", FLOOD_THRESHOLDS["min_area_sqm"] / 100)
            ).getInfo()
        except Exception as e:
            logger.warning(f"Vectorization failed: {e}")
            flood_geojson = {"features": []}

        result = {
            "event_date": event_date.isoformat(),
            "flood_area_sqkm": round(area_sqkm, 2),
            "flood_polygons": flood_geojson,
            "polygon_count": len(flood_geojson.get("features", [])),
            "source": "Sentinel-1 SAR",
            "thresholds": {
                "change_db": FLOOD_THRESHOLDS["change_threshold_db"],
                "water_db": FLOOD_THRESHOLDS["water_threshold_db"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(f"Flood detection: {area_sqkm:.2f} sq km")
        return result

    def get_latest_flood_extent(self) -> dict:
        """Get flood extent from most recent imagery."""
        return self.detect_flood_extent(datetime.utcnow())


gee_client = GEEClient()
