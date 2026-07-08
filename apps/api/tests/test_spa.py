"""SPA history-fallback tests for the static file server.

The React app uses BrowserRouter, so a refresh / bookmark / shared deep link to
``/jobs`` etc. reaches the server as a real path. ``SPAStaticFiles`` must serve
``index.html`` for those (so the client router takes over) while leaving real
assets, API routes, and ``/api`` 404s untouched.
"""

from fastapi import FastAPI
from starlette.testclient import TestClient

from taime_api.main import SPAStaticFiles, app


def test_app_boots_with_lifespan():
    """The real app starts up cleanly with the lifespan handler.

    ASGITransport (used by the async `client` fixture) does not run lifespan, so
    this is the only test that actually executes startup: init_db,
    reconcile_interrupted_jobs, and start_pool (a no-op under TAIME_DISABLE_POOL).
    """
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


def _build_app(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><html><body>SPA-SHELL</body></html>")
    (static / "assets" / "app.js").write_text("console.log('app')")

    app = FastAPI()

    @app.get("/api/v1/ping")
    def ping():
        return {"ok": True}

    app.mount("/", SPAStaticFiles(directory=str(static), html=True), name="static")
    return app


def test_spa_routes_and_assets(tmp_path):
    client = TestClient(_build_app(tmp_path))

    # Root and client-side deep links all resolve to the SPA shell.
    assert client.get("/").status_code == 200
    for route in ("/jobs", "/datasets", "/models"):
        resp = client.get(route)
        assert resp.status_code == 200
        assert "SPA-SHELL" in resp.text

    # Real assets are still served directly (not rewritten to the shell).
    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "console.log" in asset.text


def test_api_paths_are_not_masked_by_spa(tmp_path):
    client = TestClient(_build_app(tmp_path))

    # A real API route is reached, not the SPA shell.
    assert client.get("/api/v1/ping").json() == {"ok": True}

    # An unknown /api path stays a real 404 instead of returning the SPA shell.
    missing = client.get("/api/v1/does-not-exist")
    assert missing.status_code == 404
    assert "SPA-SHELL" not in missing.text
