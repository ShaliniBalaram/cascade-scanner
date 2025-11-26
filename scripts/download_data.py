"""Download precipitation and satellite data for Chennai."""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
PRECIP_DIR = DATA_DIR / "precipitation"
SAR_DIR = DATA_DIR / "sentinel1"

PRECIP_DIR.mkdir(parents=True, exist_ok=True)
SAR_DIR.mkdir(parents=True, exist_ok=True)

# Chennai bbox
CHENNAI_BBOX = {
    "north": 13.25,
    "south": 12.85,
    "east": 80.35,
    "west": 80.10,
}


def download_precipitation(days_back: int = 30) -> dict:
    """Download historical + recent precipitation from Open-Meteo."""
    print(f"Downloading {days_back} days of precipitation data...")

    # Chennai center
    lat, lon = 13.0827, 80.2707

    today = datetime.now()
    # Archive API has data up to ~5 days ago
    archive_end = today - timedelta(days=7)
    archive_start = archive_end - timedelta(days=days_back - 7)

    all_data = {"daily": {"time": [], "precipitation_sum": [], "rain_sum": []}}

    # 1. Get historical data from archive API
    print("  Fetching historical data (archive API)...")
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": archive_start.strftime("%Y-%m-%d"),
        "end_date": archive_end.strftime("%Y-%m-%d"),
        "daily": "precipitation_sum,rain_sum",
        "timezone": "Asia/Kolkata",
    }

    try:
        resp = requests.get(archive_url, params=archive_params, timeout=30)
        resp.raise_for_status()
        archive_data = resp.json()
        daily = archive_data.get("daily", {})
        all_data["daily"]["time"].extend(daily.get("time", []))
        all_data["daily"]["precipitation_sum"].extend(daily.get("precipitation_sum", []))
        all_data["daily"]["rain_sum"].extend(daily.get("rain_sum", []))
    except Exception as e:
        print(f"  Archive API warning: {e}")

    # 2. Get recent data from forecast API (has past 7 days + forecast)
    print("  Fetching recent data (forecast API)...")
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,rain_sum",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }

    try:
        resp = requests.get(forecast_url, params=forecast_params, timeout=30)
        resp.raise_for_status()
        forecast_data = resp.json()
        daily = forecast_data.get("daily", {})
        # Only add dates not already in archive data
        existing_dates = set(all_data["daily"]["time"])
        for i, d in enumerate(daily.get("time", [])):
            if d not in existing_dates:
                all_data["daily"]["time"].append(d)
                all_data["daily"]["precipitation_sum"].append(daily.get("precipitation_sum", [])[i])
                all_data["daily"]["rain_sum"].append(daily.get("rain_sum", [])[i])
    except Exception as e:
        print(f"  Forecast API warning: {e}")

    data = all_data
    start_date = archive_start
    end_date = today

    # Save JSON
    filename = f"chennai_precip_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
    filepath = PRECIP_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    # Save CSV for daily data
    csv_file = PRECIP_DIR / filename.replace(".json", "_daily.csv")
    with open(csv_file, "w") as f:
        f.write("date,precipitation_mm,rain_mm\n")
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        rain = daily.get("rain_sum", [])
        for i, d in enumerate(dates):
            f.write(f"{d},{precip[i] or 0},{rain[i] or 0}\n")

    print(f"  Saved: {filepath}")
    print(f"  Saved: {csv_file}")

    # Summary
    total_precip = sum(p or 0 for p in precip)
    max_precip = max(p or 0 for p in precip)
    rainy_days = sum(1 for p in precip if p and p > 0.1)

    return {
        "file_json": str(filepath),
        "file_csv": str(csv_file),
        "period_days": days_back,
        "total_precipitation_mm": round(total_precip, 1),
        "max_daily_mm": round(max_precip, 1),
        "rainy_days": rainy_days,
    }


def download_sentinel1_geotiff(days_back: int = 30) -> dict:
    """Download Sentinel-1 SAR data as GeoTIFF via GEE."""
    print(f"Downloading Sentinel-1 data for last {days_back} days...")

    # Initialize GEE
    key_path = os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    project_id = os.environ.get("GEE_PROJECT_ID", "cascade-scanner-gee")

    if key_path and os.path.exists(key_path):
        credentials = ee.ServiceAccountCredentials(None, key_path)
        ee.Initialize(credentials, project=project_id)
    else:
        ee.Initialize(project=project_id)

    # Chennai geometry
    bbox = ee.Geometry.Rectangle([
        CHENNAI_BBOX["west"], CHENNAI_BBOX["south"],
        CHENNAI_BBOX["east"], CHENNAI_BBOX["north"]
    ])

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Get Sentinel-1 collection
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
    )

    count = collection.size().getInfo()
    print(f"  Found {count} Sentinel-1 images")

    if count == 0:
        return {"error": "No images found", "count": 0}

    # Get latest image
    latest = collection.sort("system:time_start", False).first()

    # Get image info
    info = latest.getInfo()
    img_date = datetime.fromtimestamp(info["properties"]["system:time_start"] / 1000)

    # Download as numpy array via getDownloadURL (small area)
    # For larger areas, use Export.image.toDrive()

    try:
        # Get download URL for the region
        url = latest.getDownloadURL({
            "scale": 30,  # 30m resolution
            "region": bbox,
            "format": "GEO_TIFF",
            "bands": ["VV", "VH"],
        })

        print(f"  Downloading from GEE...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        filename = f"sentinel1_chennai_{img_date.strftime('%Y%m%d')}.tif"
        filepath = SAR_DIR / filename

        with open(filepath, "wb") as f:
            f.write(resp.content)

        print(f"  Saved: {filepath}")

        # Get file size
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        return {
            "file": str(filepath),
            "date": img_date.isoformat(),
            "size_mb": round(size_mb, 2),
            "bands": ["VV", "VH"],
            "resolution_m": 30,
            "images_available": count,
        }

    except Exception as e:
        print(f"  Direct download failed: {e}")
        print("  Trying alternative method...")

        # Alternative: sample the image and create GeoTIFF manually
        return download_sentinel1_sampled(latest, bbox, img_date, count)


def download_sentinel1_sampled(image, bbox, img_date, total_count) -> dict:
    """Download SAR data by sampling (for when direct download fails)."""

    # Sample at regular grid points
    scale = 100  # 100m sampling

    # Get VV band values
    vv_samples = image.select("VV").sampleRectangle(
        region=bbox,
        defaultValue=0
    ).getInfo()

    vh_samples = image.select("VH").sampleRectangle(
        region=bbox,
        defaultValue=0
    ).getInfo()

    vv_array = np.array(vv_samples["properties"]["VV"])
    vh_array = np.array(vh_samples["properties"]["VH"])

    # Create GeoTIFF
    height, width = vv_array.shape
    transform = from_bounds(
        CHENNAI_BBOX["west"], CHENNAI_BBOX["south"],
        CHENNAI_BBOX["east"], CHENNAI_BBOX["north"],
        width, height
    )

    filename = f"sentinel1_chennai_{img_date.strftime('%Y%m%d')}_sampled.tif"
    filepath = SAR_DIR / filename

    with rasterio.open(
        filepath, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype=vv_array.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(vv_array, 1)
        dst.write(vh_array, 2)
        dst.descriptions = ("VV", "VH")

    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"  Saved: {filepath}")

    return {
        "file": str(filepath),
        "date": img_date.isoformat(),
        "size_mb": round(size_mb, 2),
        "bands": ["VV", "VH"],
        "dimensions": f"{width}x{height}",
        "images_available": total_count,
        "method": "sampled",
    }


def main():
    """Download all data."""
    print("=" * 50)
    print("CASCADE SCANNER - DATA DOWNLOAD")
    print("=" * 50)

    # Load env
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    results = {}

    # Download precipitation
    print("\n[1/2] PRECIPITATION DATA")
    print("-" * 30)
    try:
        results["precipitation"] = download_precipitation(days_back=30)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["precipitation"] = {"error": str(e)}

    # Download Sentinel-1
    print("\n[2/2] SENTINEL-1 SAR DATA")
    print("-" * 30)
    try:
        results["sentinel1"] = download_sentinel1_geotiff(days_back=30)
    except Exception as e:
        print(f"  ERROR: {e}")
        results["sentinel1"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)

    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
