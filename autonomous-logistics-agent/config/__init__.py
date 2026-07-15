"""
config/
=======
Centralised configuration management using Pydantic Settings.

All runtime parameters — model names, API keys, retry limits,
logging format — are loaded from environment variables with
sensible defaults. This makes the system 12-factor compliant
and easy to replicate across environments.

Modules:
    - settings : AppSettings Pydantic model (single source of truth)
"""
