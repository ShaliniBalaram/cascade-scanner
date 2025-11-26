"""Neo4j database connection."""

from contextlib import contextmanager
from typing import Generator, Optional

from loguru import logger
from neo4j import GraphDatabase, Driver, Session

from src.utils.config import settings


class Neo4jConnection:
    """Neo4j connection manager (singleton)."""

    _instance: Optional["Neo4jConnection"] = None
    _driver: Optional[Driver] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._driver is None:
            self._connect()

    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
                max_connection_pool_size=settings.neo4j.max_connection_pool_size,
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {settings.neo4j.uri}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._connect()
        return self._driver

    @contextmanager
    def session(self, database: str = None) -> Generator[Session, None, None]:
        db = database or settings.neo4j.database
        session = self.driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def execute_query(self, query: str, params: dict = None) -> list:
        with self.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, params: dict = None) -> dict:
        with self.session() as session:
            result = session.run(query, params or {})
            summary = result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
            }

    def health_check(self) -> bool:
        try:
            with self.session() as session:
                result = session.run("RETURN 1 AS ok")
                return result.single()["ok"] == 1
        except Exception:
            return False

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None


neo4j_conn = Neo4jConnection()


def get_connection() -> Neo4jConnection:
    return neo4j_conn
