"""FastAPI application."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.core import CascadeScanner, query_parser, format_output
from src.core.report import report_engine, DATA_SOURCES
from src.graph import get_connection, seed_database
from src.utils.config import settings

app = FastAPI(
    title="Cascade Scanner API",
    description="Geo-AI flood-cascade risk assessment",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = CascadeScanner()


class ScanRequest(BaseModel):
    query: Optional[str] = None
    location: str = "chennai"
    time_mode: str = "nowcast"
    stakeholder: str = "emergency_manager"


class ScanResponse(BaseModel):
    event_id: str
    timestamp: str
    location: str
    risk_level: str
    alert_count: int
    safe_count: int
    summary: str
    formatted_output: str
    scan_duration_seconds: float


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        conn = get_connection()
        db_ok = conn.health_check()
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/scan", response_model=ScanResponse)
async def run_scan(request: ScanRequest):
    """Execute cascade scan."""
    try:
        # Parse NL query if provided
        if request.query:
            parsed = query_parser.parse(request.query)
            location = parsed.location
            time_mode = parsed.time_mode
            stakeholder = parsed.stakeholder
        else:
            location = request.location
            time_mode = request.time_mode
            stakeholder = request.stakeholder

        result = scanner.execute_scan(
            location=location,
            time_mode=time_mode,
            stakeholder=stakeholder,
        )

        return ScanResponse(
            event_id=result.event_id,
            timestamp=result.timestamp.isoformat(),
            location=result.location,
            risk_level=result.hazard_state.flood_risk_level,
            alert_count=len(result.alerts),
            safe_count=len(result.safe_assets),
            summary=result.summary,
            formatted_output=format_output(result, stakeholder),
            scan_duration_seconds=result.scan_duration_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan")
async def quick_scan(
    location: str = Query("chennai"),
    time_mode: str = Query("nowcast"),
    stakeholder: str = Query("emergency_manager"),
):
    """Quick scan via GET."""
    return await run_scan(ScanRequest(
        location=location,
        time_mode=time_mode,
        stakeholder=stakeholder,
    ))


@app.get("/api/v1/assets")
async def get_assets():
    """Get all assets from graph."""
    try:
        conn = get_connection()
        assets = conn.execute_query("MATCH (a:Asset) RETURN a LIMIT 100")
        return {
            "count": len(assets),
            "assets": [r["a"] for r in assets],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/seed")
async def seed_db():
    """Seed database with sample data."""
    try:
        result = seed_database()
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ REPORT ENDPOINTS ============

class ReportQuery(BaseModel):
    question: str


@app.post("/api/v1/report")
async def query_report(request: ReportQuery):
    """Answer questions about weather/hazard data with source attribution."""
    try:
        result = report_engine.query(request.question)
        return {
            "answer": result.answer,
            "data_source": {
                "name": result.data_source.name,
                "provider": result.data_source.provider,
                "url": result.data_source.url,
                "latency": result.data_source.latency,
            },
            "raw_data": result.raw_data,
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/report")
async def query_report_get(q: str = Query(..., description="Your question about weather/hazards")):
    """Answer questions via GET (e.g., ?q=was there rain last tuesday)."""
    return await query_report(ReportQuery(question=q))


@app.get("/api/v1/data-sources")
async def get_data_sources():
    """List all data sources with their characteristics."""
    return {
        "sources": {
            key: {
                "name": src.name,
                "provider": src.provider,
                "url": src.url,
                "data_type": src.data_type,
                "latency": src.latency,
                "coverage": src.coverage,
            }
            for key, src in DATA_SOURCES.items()
        }
    }


# ============ DATA DOWNLOAD ENDPOINTS ============

@app.post("/api/v1/download/precipitation")
async def download_precipitation(days: int = Query(30, ge=1, le=365)):
    """Download precipitation data for specified days."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from download_data import download_precipitation as dl_precip
        result = dl_precip(days_back=days)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/download/satellite")
async def download_satellite(days: int = Query(30, ge=1, le=90)):
    """Download Sentinel-1 SAR satellite data."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from download_data import download_sentinel1_geotiff
        result = download_sentinel1_geotiff(days_back=days)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
