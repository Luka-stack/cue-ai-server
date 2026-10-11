import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for domain errors that map to HTTP responses."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class ItemNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, item_id: str) -> None:
        super().__init__(f"Item {item_id} not found")


class InvalidItemOperationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class InvalidTimezoneError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self, tz: str) -> None:
        super().__init__(f"unknown timezone: {tz}")


# ------------------------------------------------------------------------------ chat


class ThreadNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} not found")


class ThreadBusyError(AppError):
    """Another request is already running a turn on this thread."""

    status_code = status.HTTP_409_CONFLICT

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} is busy processing another request")


class ReviewPendingError(AppError):
    """A new message was sent while the agent is waiting for a review decision."""

    status_code = status.HTTP_409_CONFLICT

    def __init__(self, thread_id: str) -> None:
        super().__init__(
            f"Thread {thread_id} has a pending memory review; accept, reject or revise it "
            "before sending another message"
        )


class NoPendingReviewError(AppError):
    status_code = status.HTTP_409_CONFLICT

    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} has no pending review to respond to")


class AgentUnavailableError(AppError):
    """The chat model (OpenAI) could not be reached or failed."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or "The chat model is unavailable, try again later")


class AgentTimeoutError(AppError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT

    def __init__(self) -> None:
        super().__init__("The chat model took too long to respond")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )
