"""Main entry point for Cascade Scanner."""

import sys
from loguru import logger


def main():
    """Run the application."""
    if len(sys.argv) < 2:
        print("Usage: python main.py [api|ui|scan|seed]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "api":
        import uvicorn
        logger.info("Starting API server...")
        uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

    elif cmd == "ui":
        import subprocess
        logger.info("Starting Streamlit UI...")
        subprocess.run(["streamlit", "run", "src/ui/app.py", "--server.port", "8501"])

    elif cmd == "scan":
        from src.core import CascadeScanner, format_output
        scanner = CascadeScanner()
        result = scanner.execute_scan()
        print(format_output(result, "emergency_manager"))

    elif cmd == "seed":
        from src.graph import seed_database
        result = seed_database()
        print(f"Seeded: {result}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
