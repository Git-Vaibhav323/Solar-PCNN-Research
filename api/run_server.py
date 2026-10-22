"""
Convenience launcher — run from the project root:

    python api/run_server.py
    python api/run_server.py --host 0.0.0.0 --port 8000
    python api/run_server.py --reload          # hot-reload for development
    python api/run_server.py --workers 4       # multiple workers (Linux/Mac only)

The server must be running before you start a JMeter test.
JMeter default target: http://localhost:8000/predict
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Solar PCNN inference server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable hot-reload (development only; do NOT use during JMeter testing)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes (>1 only works on Linux/macOS with Gunicorn)"
    )
    parser.add_argument(
        "--log-level", default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.predict:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level,
    ]

    if args.reload:
        cmd.append("--reload")
    elif args.workers > 1:
        # uvicorn multi-worker flag
        cmd += ["--workers", str(args.workers)]

    print("=" * 60)
    print(" Solar AR-PCNN Inference Server")
    print("=" * 60)
    print(f"  URL  : http://{args.host}:{args.port}")
    print(f"  Docs : http://{args.host}:{args.port}/docs")
    print(f"  Health: http://{args.host}:{args.port}/health")
    print()
    print(f"  JMeter target endpoint:")
    print(f"    POST http://{args.host}:{args.port}/predict")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    subprocess.run(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
