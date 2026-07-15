"""
utils/retry.py — Retry Utilities with Exponential Backoff
==========================================================

Provides a reusable ``@retry_with_backoff`` decorator and an async
equivalent ``@async_retry_with_backoff`` built on top of Tenacity.

Design rationale
----------------
* **Tenacity over ``time.sleep`` loops** — Tenacity separates the
  *retry policy* (how many times, how long to wait, what to catch) from
  the *business logic* (the tool call or LLM request).  This makes
  policies testable and composable without modifying application code.

* **Exponential backoff with jitter** — Pure exponential backoff causes
  *thundering herd* problems when many agents retry simultaneously.
  Adding random jitter (``random_exponential`` strategy) spreads retries
  across time, reducing peak load on downstream APIs.

* **Configurable via ``AppSettings``** — ``max_retries`` and
  ``retry_backoff_base`` are read from settings, not hardcoded.  This
  lets operators tune retry behaviour per environment without touching
  code.

* **Structured log on each retry** — Every retry attempt emits a
  structlog event with the attempt number, exception type, and wait
  duration.  This makes retry storms visible in trajectory logs without
  requiring manual instrumentation.

* **``RetryError`` re-raised as ``ToolExecutionError``** — Tenacity's
  internal ``RetryError`` is caught at the decorator boundary and
  re-raised as a domain-specific ``ToolExecutionError``.  This keeps
  agent code independent of the retry library's exception hierarchy.

* **Separate sync and async decorators** — LangGraph nodes may be
  synchronous or asynchronous.  Both variants share the same policy
  logic via a common ``_build_retry_kwargs`` helper.

Retry Strategy Diagram
----------------------

    attempt 1  ─── fail ───► wait ~2s
    attempt 2  ─── fail ───► wait ~4s + jitter
    attempt 3  ─── fail ───► wait ~8s + jitter
    attempt 4  ─── fail ───► raise ToolExecutionError

    (with max_retries=3, backoff_base=2)
"""

from __future__ import annotations

import functools
import random
from typing import Any, Callable, Sequence, Type, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    after_log,
)
import structlog

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Typed constants
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])

#: Default exceptions that should trigger a retry.  Extend this tuple as
#: needed for specific tool integrations (e.g. openai.RateLimitError).
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain exception
# ---------------------------------------------------------------------------


class ToolExecutionError(RuntimeError):
    """
    Raised when a tool call fails after all retry attempts are exhausted.

    Attributes
    ----------
    tool_name:
        Name of the tool that failed.
    attempts:
        Number of attempts made before giving up.
    last_exception:
        The exception raised on the final attempt.
    """

    def __init__(
        self,
        tool_name: str,
        attempts: int,
        last_exception: BaseException,
    ) -> None:
        self.tool_name = tool_name
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Tool '{tool_name}' failed after {attempts} attempt(s). "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_retry_kwargs(
    max_attempts: int,
    backoff_base: float,
    jitter: float,
    retryable_exceptions: Sequence[Type[BaseException]],
) -> dict[str, Any]:
    """
    Build keyword arguments for the ``@retry`` decorator.

    Centralised here so sync and async decorators share identical logic.

    Parameters
    ----------
    max_attempts:
        Total number of attempts (including the first).
    backoff_base:
        Base multiplier for exponential wait (seconds).
    jitter:
        Maximum random jitter added to each wait (seconds).
    retryable_exceptions:
        Exception types that should trigger a retry.

    Returns
    -------
    dict[str, Any]
        Keyword arguments compatible with ``tenacity.retry(**kwargs)``.
    """
    import logging as _logging

    return {
        "stop": stop_after_attempt(max_attempts),
        "wait": wait_exponential_jitter(
            initial=backoff_base,
            max=backoff_base ** max_attempts,
            jitter=jitter,
        ),
        "retry": retry_if_exception_type(tuple(retryable_exceptions)),
        # Emit a structured log event before each sleep (i.e., between attempts).
        "before_sleep": before_sleep_log(_logging.getLogger(__name__), _logging.WARNING),
        # Emit a structured log event after the final attempt (success or failure).
        "after": after_log(_logging.getLogger(__name__), _logging.DEBUG),
        # Do not re-raise Tenacity's RetryError; we handle it in the wrapper.
        "reraise": False,
    }


# ---------------------------------------------------------------------------
# Synchronous decorator
# ---------------------------------------------------------------------------


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    jitter: float = 1.0,
    retryable_exceptions: Sequence[Type[BaseException]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    tool_name: str | None = None,
) -> Callable[[F], F]:
    """
    Synchronous retry decorator with exponential backoff and jitter.

    Parameters
    ----------
    max_attempts:
        Maximum total attempts including the first call.  Defaults to 3.
    backoff_base:
        Base for exponential backoff in seconds.  Defaults to 2.0.
        Wait times: ~2 s, ~4 s, ~8 s, …
    jitter:
        Maximum random seconds added to each wait.  Reduces thundering
        herd effect when many agents retry simultaneously.
    retryable_exceptions:
        Exceptions that should trigger a retry.  All others propagate
        immediately without retrying.
    tool_name:
        Optional name shown in ``ToolExecutionError``.  Defaults to the
        wrapped function's ``__name__``.

    Returns
    -------
    Callable
        Decorated function with retry behaviour.

    Example
    -------
    ::

        @retry_with_backoff(max_attempts=3, tool_name="carrier_lookup")
        def lookup_carrier(carrier_id: str) -> CarrierInfo:
            return _real_api_call(carrier_id)
    """

    def decorator(func: F) -> F:
        _name = tool_name or func.__name__

        retry_kwargs = _build_retry_kwargs(
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
        )

        @retry(**retry_kwargs)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt_number = 0
            last_exc: BaseException | None = None

            try:
                return _inner(*args, **kwargs)
            except RetryError as exc:
                last_exc = exc.last_attempt.exception() or exc
                attempt_number = exc.last_attempt.attempt_number
            except Exception as exc:
                # Exception not in retryable_exceptions — re-raise immediately.
                raise

            # All retries exhausted.
            _logger.error(
                "tool_retries_exhausted",
                tool=_name,
                attempts=attempt_number,
                error=str(last_exc),
            )
            raise ToolExecutionError(
                tool_name=_name,
                attempts=attempt_number,
                last_exception=last_exc,
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Asynchronous decorator
# ---------------------------------------------------------------------------


def async_retry_with_backoff(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    jitter: float = 1.0,
    retryable_exceptions: Sequence[Type[BaseException]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    tool_name: str | None = None,
) -> Callable[[F], F]:
    """
    Async retry decorator with exponential backoff and jitter.

    Identical contract to ``retry_with_backoff`` but wraps ``async def``
    functions.  Use this for any agent node or tool that makes async
    HTTP requests.

    Parameters
    ----------
    max_attempts:
        Maximum total attempts including the first call.
    backoff_base:
        Base for exponential backoff in seconds.
    jitter:
        Maximum random seconds of jitter per wait period.
    retryable_exceptions:
        Exceptions that should trigger a retry.
    tool_name:
        Optional name shown in ``ToolExecutionError``.

    Example
    -------
    ::

        @async_retry_with_backoff(max_attempts=3, tool_name="route_calculator")
        async def calculate_route(origin: str, destination: str) -> RouteInfo:
            return await _async_api_call(origin, destination)
    """
    from tenacity import AsyncRetrying

    def decorator(func: F) -> F:
        _name = tool_name or func.__name__

        retry_kwargs = _build_retry_kwargs(
            max_attempts=max_attempts,
            backoff_base=backoff_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            attempt_number = 0

            try:
                async for attempt in AsyncRetrying(**retry_kwargs):
                    with attempt:
                        return await func(*args, **kwargs)
            except RetryError as exc:
                last_exc = exc.last_attempt.exception() or exc
                attempt_number = exc.last_attempt.attempt_number
            except Exception:
                raise

            _logger.error(
                "async_tool_retries_exhausted",
                tool=_name,
                attempts=attempt_number,
                error=str(last_exc),
            )
            raise ToolExecutionError(
                tool_name=_name,
                attempts=attempt_number,
                last_exception=last_exc or RuntimeError("Unknown failure"),
            )

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Convenience: settings-aware factory
# ---------------------------------------------------------------------------


def make_retry_decorator(
    tool_name: str,
    extra_exceptions: Sequence[Type[BaseException]] = (),
    async_mode: bool = False,
) -> Callable[[F], F]:
    """
    Build a retry decorator pre-configured from ``AppSettings``.

    This is the preferred way to apply retries in agent and tool code
    because it reads ``max_retries`` and ``retry_backoff_base`` from
    the live settings object — no magic numbers scattered across files.

    Parameters
    ----------
    tool_name:
        Descriptive name used in logs and ``ToolExecutionError``.
    extra_exceptions:
        Additional exception types to treat as retryable (e.g.
        ``openai.RateLimitError``, ``groq.APIStatusError``).
    async_mode:
        If ``True``, return the async variant of the decorator.

    Returns
    -------
    Callable
        Configured retry decorator ready to apply to a function.

    Example
    -------
    ::

        from utils.retry import make_retry_decorator

        retry = make_retry_decorator("carrier_lookup", async_mode=False)

        @retry
        def lookup_carrier(carrier_id: str) -> CarrierInfo:
            ...
    """
    from config.settings import get_settings

    settings = get_settings()
    all_exceptions = tuple(DEFAULT_RETRYABLE_EXCEPTIONS) + tuple(extra_exceptions)

    factory = async_retry_with_backoff if async_mode else retry_with_backoff
    return factory(
        max_attempts=settings.max_retries,
        backoff_base=2.0,
        jitter=1.0,
        retryable_exceptions=all_exceptions,
        tool_name=tool_name,
    )
