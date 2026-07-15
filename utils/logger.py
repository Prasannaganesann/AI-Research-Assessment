"""
utils/logger.py — Structured Logging via Structlog
====================================================

Provides a single ``configure_logging()`` call that sets up Structlog
with either a human-readable console renderer (development) or a
machine-parseable JSON renderer (production / CI).

Design rationale
----------------
* **Structlog over stdlib ``logging``** — Structlog encourages
  *key=value* style log entries rather than format-string messages.
  Every event carries structured context that can be indexed by
  log aggregators (Datadog, CloudWatch, Loki) without regex parsing.
  This is essential when debugging multi-agent trajectory failures.

* **stdlib integration via ``ProcessorFormatter``** — Third-party
  libraries (LangChain, httpx) emit stdlib ``logging`` records.
  Structlog's ``ProcessorFormatter`` intercepts these and renders them
  through the same pipeline as native structlog events.  One log
  stream, one format, zero surprise.

* **``timestamper`` processor** — ISO 8601 timestamps with timezone
  are added to every event.  UTC is used regardless of the server's
  local timezone to avoid ambiguity in distributed systems.

* **``add_log_level`` + ``add_logger_name`` processors** — These add
  ``level`` and ``logger`` keys to every event dict, making it trivial
  to filter logs by severity or origin module in any log viewer.

* **``StackInfoRenderer`` + ``ExceptionRenderer``** — Exceptions are
  serialised as structured dicts (not free-text tracebacks) in JSON
  mode.  In console mode they render as colour-highlighted tracebacks.

* **``get_logger(__name__)`` pattern** — Every module calls
  ``get_logger(__name__)`` to obtain a bound logger.  The module name
  is embedded in every log event, so trajectory failures are
  immediately traceable to the exact agent or tool that caused them.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    fmt: Literal["json", "console"] = "console",
) -> None:
    """
    Initialise Structlog and stdlib logging in one call.

    Must be called exactly once at application startup (``main.py``)
    before any module emits a log event.  Subsequent calls are
    idempotent but will reconfigure the pipeline — avoid in hot paths.

    Parameters
    ----------
    level:
        Minimum log level string, e.g. ``"DEBUG"``, ``"INFO"``.
        Case-insensitive.
    fmt:
        ``"console"`` → ColourfulConsoleRenderer (development)
        ``"json"``    → JSONRenderer (production / CI)
    """
    _numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ------------------------------------------------------------------
    # Shared processors applied to every log event (both structlog and
    # stdlib records captured via ProcessorFormatter).
    # ------------------------------------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        # Inject the log level ("info", "error", …) into every event.
        structlog.stdlib.add_log_level,
        # Inject the logger name (module path) into every event.
        structlog.stdlib.add_logger_name,
        # ISO 8601 timestamp in UTC on every event.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Merge extra ``positional_args`` added by stdlib adapters.
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Expand ``stack_info=True`` argument into a readable string.
        structlog.processors.StackInfoRenderer(),
        # Format exception info attached to the event.
        structlog.processors.format_exc_info,
        # Decode byte strings in the event dict.
        structlog.processors.UnicodeDecoder(),
    ]

    # ------------------------------------------------------------------
    # Final renderer: JSON for production, colourful console for dev.
    # ------------------------------------------------------------------
    if fmt == "json":
        final_renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        final_renderer = structlog.dev.ConsoleRenderer(colors=True)

    # ------------------------------------------------------------------
    # Configure Structlog itself.
    # ------------------------------------------------------------------
    structlog.configure(
        processors=shared_processors
        + [
            # Bridge: allows stdlib loggers to be routed through structlog.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ------------------------------------------------------------------
    # Configure the stdlib root logger so third-party libraries (e.g.
    # httpx, openai) are also rendered through our pipeline.
    # ------------------------------------------------------------------
    formatter = structlog.stdlib.ProcessorFormatter(
        # These processors run only on stdlib records (not native structlog).
        foreign_pre_chain=shared_processors,
        # Final processor runs on ALL records.
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(_numeric_level)

    # Silence noisy third-party loggers that are not useful for this project.
    _silence = [
        "httpx",
        "httpcore",
        "openai._base_client",
        "groq._base_client",
        "urllib3.connectionpool",
    ]
    for name in _silence:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Usage (in any module)
    ---------------------
    ::

        from utils.logger import get_logger

        logger = get_logger(__name__)

        logger.info("shipment_rerouted", carrier="FedEx", route_id="R-42")
        logger.warning("tool_retry", attempt=2, tool="carrier_lookup")
        logger.error("graph_halt", reason="max_steps_exceeded", steps=20)

    The ``__name__`` pattern ensures the ``logger`` key in every log
    event carries the full module path (e.g. ``agents.planner_agent``),
    making it trivial to filter events by origin in any log viewer.

    Parameters
    ----------
    name:
        Module name — always pass ``__name__``.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Bound logger instance with structured key-value API.
    """
    return structlog.get_logger(name)
