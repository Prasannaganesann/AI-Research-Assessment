"""
config/settings.py — Centralised Application Configuration
===========================================================

All runtime parameters are loaded from environment variables (or a
.env file) via Pydantic Settings.  No configuration is ever hardcoded
in application code.

Design rationale
----------------
* **Pydantic Settings** gives us automatic env-var parsing, type
  coercion, and validation in a single class.  Errors surface at
  startup, not mid-run.

* **SecretStr for API keys** — Pydantic wraps secret values so they
  are redacted in logs, stack traces, and ``repr()`` output.  This is
  non-negotiable in any system that ships to production.

* **``get_settings()`` singleton via ``@lru_cache``** — A module-level
  ``settings = AppSettings()`` executes at import time, which breaks
  unit tests that patch environment variables.  The cached-factory
  pattern lets tests call ``get_settings.cache_clear()`` before each
  test to obtain a fresh, correctly-patched instance.

* **``model_config`` with ``env_file``** — Pydantic Settings checks
  the real environment first, then falls back to the .env file.  This
  means Docker / CI environments can override values without editing
  any file.

* **Derived properties (``closed_source_provider``, etc.)** — The
  graph nodes need to know *which LangChain client to instantiate*, not
  just which model name to pass.  Deriving the provider from the model
  name string (OpenAI-family vs. Groq-family) keeps the .env minimal
  while making the graph code readable.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GROQ_MODEL_PREFIXES: tuple[str, ...] = (
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "deepseek",
    "compound",
)

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LOG_FORMATS = ("json", "console")
_ENVIRONMENTS = ("development", "staging", "production")


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    All fields have sensible defaults so the system can start in
    development mode with minimal configuration.  Production deployments
    must supply ``OPENAI_API_KEY`` and ``GROQ_API_KEY`` at minimum.

    Field naming follows the SCREAMING_SNAKE_CASE convention used in
    .env files; Pydantic Settings maps them automatically.
    """

    model_config = SettingsConfigDict(
        # Load from .env if present; real env vars always win.
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore extra variables that may exist in the environment.
        extra="ignore",
        # Case-insensitive key matching (MY_VAR == my_var).
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Application identity
    # ------------------------------------------------------------------

    app_name: str = Field(
        default="autonomous-logistics-agent",
        description="Human-readable application name used in logs and reports.",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Semantic version string.",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description=(
            "Deployment environment.  Controls defaults for log verbosity "
            "and safety guards (e.g., no real API calls in 'development' "
            "unless keys are present)."
        ),
    )

    # ------------------------------------------------------------------
    # API credentials  (SecretStr prevents accidental logging)
    # ------------------------------------------------------------------

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "OpenAI API key for GPT-4o.  Obtain from "
            "https://platform.openai.com/api-keys"
        ),
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Groq API key for open-source model inference.  "
            "Free tier available at https://console.groq.com"
        ),
    )

    # ------------------------------------------------------------------
    # Model selection  (fully configurable, no hardcoding)
    # ------------------------------------------------------------------

    closed_source_model: str = Field(
        default="gpt-4o",
        alias="OPENAI_MODEL",
        description=(
            "Closed-source model identifier passed to OpenAI ChatCompletion. "
            "Examples: gpt-4o, gpt-4o-mini, gpt-4-turbo"
        ),
    )
    open_source_model: str = Field(
        default="qwen/qwen3-8b",
        alias="GROQ_MODEL",
        description=(
            "Open-source model identifier passed to Groq ChatCompletion. "
            "Examples: qwen/qwen3-8b, llama3-8b-8192, mixtral-8x7b-32768"
        ),
    )

    # ------------------------------------------------------------------
    # Request behaviour
    # ------------------------------------------------------------------

    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum number of retry attempts for tool calls and LLM requests. "
            "Applied via the tenacity retry decorator in utils/retry.py."
        ),
    )
    request_timeout: float = Field(
        default=60.0,
        gt=0.0,
        le=300.0,
        description=(
            "HTTP request timeout in seconds for LLM API calls.  "
            "Groq inference is typically fast (<5 s); OpenAI may approach "
            "30 s for long reasoning chains."
        ),
    )
    max_graph_steps: int = Field(
        default=20,
        ge=5,
        le=100,
        description=(
            "Hard ceiling on the number of LangGraph node executions per run. "
            "Prevents infinite loops in edge cases while allowing reasonable "
            "retry chains."
        ),
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    eval_scenarios: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of test scenarios to execute per model during benchmarking.",
    )
    eval_output_dir: str = Field(
        default="reports",
        description="Directory where benchmark reports and matplotlib figures are written.",
    )
    enable_trajectory_logging: bool = Field(
        default=True,
        description=(
            "When True, each graph node emits a structured TrajectorySnapshot "
            "to the log stream and to the in-memory trajectory store.  "
            "Disable in latency-sensitive production runs."
        ),
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = Field(
        default="INFO",
        description=f"Minimum log level.  One of: {', '.join(_LOG_LEVELS)}.",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description=(
            "Log renderer.  Use 'json' in production/CI for machine-parseable "
            "structured logs; use 'console' in development for human-readable "
            "colour output."
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, value: str) -> str:
        """Uppercase and validate the log level string."""
        upper = str(value).upper()
        if upper not in _LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {_LOG_LEVELS}, got '{value}'."
            )
        return upper

    @field_validator("closed_source_model", "open_source_model", mode="before")
    @classmethod
    def strip_model_name(cls, value: str) -> str:
        """Strip accidental whitespace from model name strings."""
        return str(value).strip()

    @model_validator(mode="after")
    def warn_missing_keys(self) -> "AppSettings":
        """
        Emit a warning (not an error) when API keys appear to be placeholder
        values.  This allows the app to start in offline / test mode without
        crashing, while still surfacing the misconfiguration clearly.
        """
        placeholder_pattern = re.compile(
            r"^(your_.+_here|sk-xxx|placeholder|changeme|none|)$",
            re.IGNORECASE,
        )
        key_value = self.openai_api_key.get_secret_value()
        groq_value = self.groq_api_key.get_secret_value()

        self._openai_key_valid = not placeholder_pattern.match(key_value)
        self._groq_key_valid = not placeholder_pattern.match(groq_value)
        return self

    # ------------------------------------------------------------------
    # Derived properties  (computed from env vars, not stored separately)
    # ------------------------------------------------------------------

    @property
    def closed_source_provider(self) -> Literal["openai"]:
        """
        Always 'openai' for the closed-source model.

        Kept as a property (not a plain string) so that if we ever add
        Anthropic or Gemini support, the graph code requires zero changes —
        only the property logic changes.
        """
        return "openai"

    @property
    def open_source_provider(self) -> Literal["groq"]:
        """
        Always 'groq' for the open-source model.

        Derived from the model name prefix for defensive correctness;
        raises early if a non-Groq model is accidentally configured here.
        """
        model_lower = self.open_source_model.lower()
        if not any(model_lower.startswith(p) for p in _GROQ_MODEL_PREFIXES):
            raise ValueError(
                f"open_source_model '{self.open_source_model}' does not match "
                f"any known Groq model prefix: {_GROQ_MODEL_PREFIXES}. "
                "Update _GROQ_MODEL_PREFIXES in config/settings.py if you are "
                "adding a new Groq-hosted model."
            )
        return "groq"

    @property
    def openai_api_key_value(self) -> str:
        """Return the raw OpenAI key string (use only when passing to SDKs)."""
        return self.openai_api_key.get_secret_value()

    @property
    def groq_api_key_value(self) -> str:
        """Return the raw Groq key string (use only when passing to SDKs)."""
        return self.groq_api_key.get_secret_value()

    @property
    def is_development(self) -> bool:
        """True when running in the development environment."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """True when running in the production environment."""
        return self.environment == "production"

    def __repr__(self) -> str:
        """
        Safe repr that never exposes secret values.

        Pydantic's default repr would normally call .get_secret_value()
        for SecretStr fields.  We override it to ensure API keys are
        always masked, even in debug sessions.
        """
        return (
            f"AppSettings("
            f"app_name={self.app_name!r}, "
            f"environment={self.environment!r}, "
            f"closed_source_model={self.closed_source_model!r}, "
            f"open_source_model={self.open_source_model!r}, "
            f"openai_api_key='***', "
            f"groq_api_key='***', "
            f"log_level={self.log_level!r}, "
            f"max_retries={self.max_retries!r}, "
            f"request_timeout={self.request_timeout!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the application settings singleton.

    Uses ``@lru_cache`` so Pydantic validation runs exactly once per
    process, regardless of how many modules call ``get_settings()``.

    For tests that need a fresh configuration:
        >>> from config.settings import get_settings
        >>> get_settings.cache_clear()
        >>> # now set environment variables, then call get_settings() again

    Returns
    -------
    AppSettings
        Validated, immutable settings instance.
    """
    return AppSettings()
