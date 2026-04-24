# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Parkinsons Motor Environment.

This module creates an HTTP server that exposes the ParkinsonsMotorEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from server.parkinsons_Motor_environment import ParkinsonsMotorEnvironment
except ImportError:
    from parkinsons_Motor.core.models import ParkinsonsMotorAction, ParkinsonsMotorObservation
    from parkinsons_Motor.server.parkinsons_Motor_environment import (
        ParkinsonsMotorEnvironment,
    )


# Create the app with web interface and README integration
app = create_app(
    ParkinsonsMotorEnvironment,
    ParkinsonsMotorAction,
    ParkinsonsMotorObservation,
    env_name="parkinsons_Motor",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# Mount static assets and expose MyoSuite demo at /viewer
_static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


_viewer_html_path = _static_dir / "myosuite_demo" / "index.html"


@app.get("/viewer", response_class=HTMLResponse)
async def viewer():
    html = _viewer_html_path.read_text(encoding="utf-8")
    # Inject base tag so all relative paths resolve against the static mount
    html = html.replace(
        "<head>",
        '<head>\n        <base href="/static/myosuite_demo/">',
        1,
    )
    return HTMLResponse(content=html)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m parkinsons_Motor.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn parkinsons_Motor.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 8000:
        main()
    else:
        main(host=args.host, port=args.port)
