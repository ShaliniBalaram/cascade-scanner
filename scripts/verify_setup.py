"""Verify POC setup is complete."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_python():
    print("Python version...", end=" ")
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        print(f"OK ({v.major}.{v.minor}.{v.micro})")
        return True
    print(f"FAIL ({v.major}.{v.minor})")
    return False


def check_deps():
    print("\nDependencies:")
    deps = [
        ("fastapi", "fastapi"),
        ("neo4j", "neo4j"),
        ("geopandas", "geopandas"),
        ("streamlit", "streamlit"),
        ("scikit-learn", "sklearn"),
        ("loguru", "loguru"),
        ("earthengine-api", "ee"),
    ]
    ok = True
    for name, mod in deps:
        try:
            __import__(mod)
            print(f"  [x] {name}")
        except ImportError:
            print(f"  [ ] {name} MISSING")
            ok = False
    return ok


def check_dirs():
    print("\nDirectories:")
    dirs = ["src/api", "src/core", "src/graph", "src/ml", "src/ui", "config", "data"]
    ok = True
    for d in dirs:
        path = project_root / d
        if path.exists():
            print(f"  [x] {d}")
        else:
            print(f"  [ ] {d} MISSING")
            ok = False
    return ok


def check_configs():
    print("\nConfig files:")
    files = [
        ".env",
        "config/environments/development.yaml",
        "config/assets/chennai_assets.yaml",
        "config/fragility_curves/chennai_v1.yaml",
    ]
    ok = True
    for f in files:
        path = project_root / f
        if path.exists():
            print(f"  [x] {f}")
        else:
            print(f"  [ ] {f} MISSING")
            ok = False
    return ok


def check_neo4j():
    print("\nNeo4j connection...", end=" ")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687", auth=("neo4j", "cascade_scanner_2024")
        )
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_postgres():
    print("PostgreSQL connection...", end=" ")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", port=5432,
            user="cascade_user", password="cascade_pass_2024", database="cascade_cache"
        )
        conn.close()
        print("OK")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    print("=" * 50)
    print("CASCADE SCANNER - SETUP VERIFICATION")
    print("=" * 50)

    results = [
        check_python(),
        check_deps(),
        check_dirs(),
        check_configs(),
    ]

    # Optional DB checks
    print("\nDatabase (requires Docker):")
    check_neo4j()
    check_postgres()

    print("\n" + "=" * 50)
    if all(results):
        print("Core setup: COMPLETE")
        print("Start Docker for database, then run Part 2")
    else:
        print("Some checks failed. Review above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
