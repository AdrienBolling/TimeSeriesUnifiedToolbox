"""Structured logging for the TSUT Framework.

Public API
----------
- :class:`Logger`        - structured logger (wraps :class:`logging.Logger`).
- :class:`JSONFormatter` - :class:`logging.Formatter` that emits single-line JSON.
- :class:`TextFormatter` - :class:`logging.Formatter` for human-readable terminal output.
- :func:`configure`      - one-call setup for the ``tsut`` logging hierarchy.
"""

from tsut.core.common.logging.logger import (
    JSONFormatter,
    Logger,
    TextFormatter,
    configure,
)

__all__ = ["Logger", "JSONFormatter", "TextFormatter", "configure"]
