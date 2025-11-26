"""Seed Neo4j Aura database with Chennai assets."""
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

def seed_aura(uri: str, user: str, password: str):
    """Seed Aura database."""
    from neo4j import GraphDatabase
    
    print(f"Connecting to {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Test connection
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        print("Connected successfully!")
    
    # Now set environment and seed
    os.environ["NEO4J_URI"] = uri
    os.environ["NEO4J_USER"] = user
    os.environ["NEO4J_PASSWORD"] = password
    
    from src.graph.seed import seed_database
    result = seed_database()
    print(f"Seeded: {result}")
    
    driver.close()
    return result

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python seed_aura.py <uri> <user> <password>")
        print("Example: python seed_aura.py neo4j+s://xxx.databases.neo4j.io neo4j mypassword")
        sys.exit(1)
    
    seed_aura(sys.argv[1], sys.argv[2], sys.argv[3])
