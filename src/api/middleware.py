"""Middleware de la API: logging, timing, CORS, error handling global."""
from __future__ import annotations

import time
import traceback
import uuid

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.schemas import ErrorResponse
from src.config.logging import get_logger
from src.config.settings import get_settings

log = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log estructurado de cada request con timing y request_id."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        log.info("request_start", method=request.method, path=request.url.path, request_id=request_id)

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            log.error("request_unhandled_exception", path=request.url.path, error=str(exc), elapsed_ms=elapsed)
            raise

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        log.info("request_complete", method=request.method, path=request.url.path,
                 status=response.status_code, elapsed_ms=elapsed, request_id=request_id)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed}ms"
        return response


def _status_to_code(http_status: int) -> str:
    return {
        400: "bad_request", 401: "unauthenticated", 403: "forbidden",
        404: "not_found", 409: "conflict", 410: "gone",
        413: "payload_too_large", 422: "validation_error",
        429: "rate_limit_exceeded", 500: "internal_error", 503: "service_unavailable",
    }.get(http_status, "error")


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    from fastapi import HTTPException  # noqa: PLC0415
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=_status_to_code(exc.status_code),
                detail=str(exc.detail),
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )
    return await unhandled_exception_handler(request, exc)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    log.error("unhandled_exception", path=request.url.path, error=str(exc),
              traceback=traceback.format_exc()[-2000:], request_id=request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="Error interno del servidor.",
            request_id=request_id,
        ).model_dump(),
    )


def register_middleware(app: FastAPI) -> None:
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )
    app.add_middleware(RequestLoggingMiddleware)


def register_exception_handlers(app: FastAPI) -> None:
    from fastapi import HTTPException  # noqa: PLC0415
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
