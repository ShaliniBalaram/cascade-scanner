"""Configuration loader for Cascade Scanner."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "cascade_scanner_2024"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "cascade_user"
    password: str = "cascade_pass_2024"
    database: str = "cascade_cache"
    pool_size: int = 10

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class GEEConfig(BaseModel):
    project_id: Optional[str] = None
    service_account_key: Optional[str] = None
    cache_ttl_days: int = 7


class IMDConfig(BaseModel):
    base_url: str = "https://rmc.imd.gov.in/"
    cache_ttl_hours: int = 1
    timeout_seconds: int = 30
    retry_attempts: int = 3


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1
    cors_origins: list[str] = ["http://localhost:8501"]


class UIConfig(BaseModel):
    port: int = 8501
    theme: str = "light"
    map_center: dict = {"lat": 13.0827, "lon": 80.2707}
    map_zoom: int = 11


class BoundingBox(BaseModel):
    north: float = 13.25
    south: float = 12.85
    east: float = 80.35
    west: float = 80.05


class ChennaiConfig(BaseModel):
    name: str = "Chennai Metropolitan Area"
    bbox: BoundingBox = BoundingBox()
    center: dict = {"lat": 13.0827, "lon": 80.2707}
    timezone: str = "Asia/Kolkata"


class ScannerConfig(BaseModel):
    default_time_mode: str = "nowcast"
    alert_threshold_probability: float = 0.75
    max_results: int = 50
    timeout_seconds: int = 30


class LoggingConfig(BaseModel):
    level: str = "DEBUG"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "7 days"


class AppConfig(BaseModel):
    name: str = "cascade_scanner"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    postgres: PostgresConfig = PostgresConfig()
    gee: GEEConfig = GEEConfig()
    imd: IMDConfig = IMDConfig()
    api: APIConfig = APIConfig()
    ui: UIConfig = UIConfig()
    scanner: ScannerConfig = ScannerConfig()
    chennai: ChennaiConfig = ChennaiConfig()


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def load_yaml_config(env: str = "development") -> dict[str, Any]:
    config_path = get_project_root() / "config" / "environments" / f"{env}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_settings(env: Optional[str] = None) -> Settings:
    env = env or os.getenv("APP_ENV", "development")
    yaml_config = load_yaml_config(env)

    # Override with env vars
    if os.getenv("NEO4J_URI"):
        yaml_config.setdefault("neo4j", {})["uri"] = os.getenv("NEO4J_URI")
    if os.getenv("NEO4J_PASSWORD"):
        yaml_config.setdefault("neo4j", {})["password"] = os.getenv("NEO4J_PASSWORD")
    if os.getenv("GEE_PROJECT_ID"):
        yaml_config.setdefault("gee", {})["project_id"] = os.getenv("GEE_PROJECT_ID")
    if os.getenv("GEE_SERVICE_ACCOUNT_KEY"):
        yaml_config.setdefault("gee", {})["service_account_key"] = os.getenv("GEE_SERVICE_ACCOUNT_KEY")

    return Settings(**yaml_config) if yaml_config else Settings()


settings = get_settings()
