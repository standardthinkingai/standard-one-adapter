"""Standalone HTTP process; only the backend client talks to an inference server."""

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .backend import AdapterError, ScoringBackend
from .protocol import SystemOneRequest
from .service import SystemOneService

logger = logging.getLogger(__name__)


async def score_http(service, body, request):
    """Share disconnect cancellation between standalone and integrated serving."""
    work = asyncio.create_task(service.score(body))
    stop_watching = False

    async def disconnected():
        while not stop_watching:
            if await request.is_disconnected():
                return
            if not stop_watching:
                await asyncio.sleep(0.05)

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (work, watcher), return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return await work
        raise AdapterError("client_disconnected", "Client disconnected.", status=499)
    except AdapterError:
        raise
    except Exception:
        logger.exception("Decision scoring failed")
        raise AdapterError(
            "scoring_failed", "Could not compute a complete decision.", status=500
        ) from None
    finally:
        stop_watching = True
        work.cancel()
        watcher.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


def create_app(
    backend: ScoringBackend,
    *,
    api_key: str | None = None,
    max_concurrency: int = 32,
    alias: str = "jev-latest",
    default_temperature: float = 1.0,
    temperature_by_type: dict[str, float] | None = None,
    prompt_wording: str = "served",
) -> FastAPI:
    service = SystemOneService(
        backend,
        max_concurrency=max_concurrency,
        alias=alias,
        default_temperature=default_temperature,
        temperature_by_type=temperature_by_type,
        prompt_wording=prompt_wording,
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            await backend.start()
            yield
        finally:
            await backend.close()

    app = FastAPI(title="Jev adapter", version="0.1.0", lifespan=lifespan)

    async def authorize(request: Request):
        if api_key and not secrets.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            f"Bearer {api_key}".encode(),
        ):
            raise AdapterError(
                "unauthorized", "A valid bearer token is required.", status=401
            )

    @app.exception_handler(AdapterError)
    async def adapter_error(request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "field": exc.field,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": error["msg"],
                    "field": ".".join(str(p) for p in error["loc"] if p != "body"),
                }
            },
        )

    @app.get("/health")
    async def health():
        # Process liveness, not an inference/GPU readiness probe.
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": backend.model,
                    "object": "model",
                    "owned_by": "inference-engine",
                    **(
                        {"label_scheme": backend.label_scheme}
                        if getattr(backend, "label_scheme", None)
                        else {}
                    ),
                },
            ],
        }

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    async def systemone(body: SystemOneRequest, request: Request):
        return await score_http(service, body, request)

    return app
