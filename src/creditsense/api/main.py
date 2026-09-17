import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from starlette.routing import Route

from creditsense.api.applications import router as applications_router
from creditsense.api.browse import router as browse_router
from creditsense.api.intake import router as intake_router
from creditsense.api.predict import router as prediction_router
from creditsense.api.stream import router as stream_router
from creditsense.config import get_settings
from creditsense.logging_config import configure_logging, get_logger
from creditsense.mcp_server.server import build_server

_FRONTEND_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "static"

configure_logging()
logger = get_logger(__name__)

settings = get_settings()
mcp_server = build_server()
_public_url = urlparse(settings.mcp_public_base_url)
_public_netloc = _public_url.netloc or "localhost:8000"
_public_hostname = _public_url.hostname or "localhost"

# streamable_http_path="/" -- the SDK's default internal path is itself "/mcp", which
# would combine with FastAPI's app.mount("/mcp", ...) below into "/mcp/mcp". Mounting
# at "/" here makes the final route exactly "/mcp".
#
# transport_security is passed EXPLICITLY, not left to default: the SDK auto-enables
# DNS-rebinding protection allowing only 127.0.0.1/localhost/[::1] whenever `host` is
# left at its "127.0.0.1" default, so any request arriving with a different Host header
# -- exactly what happens behind a tunnel, where Claude Desktop's connector sends
# Host: <something>.ngrok-free.app -- is rejected with 421 Misdirected Request before
# it ever reaches auth. The fix is to allow-list the configured public host, NOT to
# switch the protection off.
mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*", "localhost:*", "[::1]:*", "localhost", "127.0.0.1",
            _public_netloc, _public_hostname, f"{_public_hostname}:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
            f"{_public_url.scheme or 'http'}://{_public_netloc}",
            f"https://{_public_hostname}", f"https://{_public_hostname}:*",
            "https://claude.ai",
        ],
    ),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Mounting streamable_http_app() alone does NOT start the MCP session manager --
    # the SDK's own docstring on MCPServer.session_manager says it raises
    # RuntimeError if accessed before streamable_http_app() has been called, and the
    # manager still has to be explicitly run for the mounted app to handle sessions
    # at all. This is the one wiring step that fails at request time, not at import,
    # if skipped.
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(prediction_router)
app.include_router(applications_router)
app.include_router(browse_router)
app.include_router(stream_router)
app.include_router(intake_router)
# Mounted before the request-ID middleware is added below, so middleware added via
# add_middleware() still wraps traffic routed to /mcp -- deliberate (P7.3 wants
# request IDs everywhere), but the middleware itself must never read the request
# body, since MCP's streamable-HTTP transport uses SSE and a consumed body breaks it.
app.mount("/mcp", mcp_app)

# OAuth discovery must answer at the ORIGIN ROOT, not under /mcp. The SDK builds its
# .well-known routes inside mcp_app, which lands them at /mcp/.well-known/... once
# mounted -- but RFC 8414/9728 clients (Claude Desktop's connector among them) look
# for a path-aware issuer's metadata at <origin>/.well-known/<endpoint>/<issuer path>,
# i.e. /.well-known/oauth-authorization-server/mcp. Without this re-registration those
# lookups 404 and the whole connector flow fails before it starts. Verified by request,
# not assumed -- see tests/test_mcp_oauth_discovery.py.
for _mcp_route in mcp_app.routes:
    _path = getattr(_mcp_route, "path", "")
    if not _path.startswith("/.well-known/"):
        continue
    _methods = sorted(getattr(_mcp_route, "methods", None) or {"GET"})
    # The authorization-server document is served both path-aware (what RFC 8414 says
    # for an issuer with a path) and bare (what simpler clients try first).
    _targets = {_path} if _path.endswith("/mcp") else {f"{_path}/mcp", _path}
    for _target in _targets:
        app.router.routes.append(Route(_target, endpoint=_mcp_route.app, methods=_methods))

# Mounted at /ui, not /, so it can never shadow the API routes above regardless of
# registration order (a Mount at "/" would need to be registered strictly last to act
# only as a catch-all -- mounting at a dedicated prefix sidesteps that ordering trap
# entirely). P9.1's frontend: plain static HTML/JS/Tailwind-CDN, no build step.
app.mount("/ui", StaticFiles(directory=_FRONTEND_STATIC_DIR, html=True), name="ui")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred."},
        headers={"X-Request-ID": request.headers.get("X-Request-ID", "")},
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}
