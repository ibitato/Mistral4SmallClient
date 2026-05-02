"""Logging configuration helpers for the dual-model Mistral CLI."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DEFAULT_LOG_RETENTION_DAYS = 2
DEFAULT_LOG_FILE_NAME = "mistralcli.log"

ENV_LOG_DIR = "MISTRAL_LOCAL_LOG_DIR"
ENV_LOG_RETENTION_DAYS = "MISTRAL_LOCAL_LOG_RETENTION_DAYS"
ENV_DEBUG = "MISTRAL_LOCAL_DEBUG"

LOGGER_NAME = "mistralcli"
_MANAGED_HANDLER_ATTR = "_mistralcli_managed"


def _default_log_directory() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "mistralcli" / "logs"
    return Path.home() / ".local" / "state" / "mistralcli" / "logs"


def _parse_bool(raw_value: str | None, *, default: bool) -> bool:
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Runtime logging configuration for the CLI."""

    directory: Path = field(default_factory=_default_log_directory)
    debug_enabled: bool = True
    retention_days: int = DEFAULT_LOG_RETENTION_DAYS
    file_name: str = DEFAULT_LOG_FILE_NAME

    @classmethod
    def from_env(cls) -> LoggingConfig:
        """Build logging defaults from environment variables."""

        configured_dir = os.environ.get(ENV_LOG_DIR, "").strip()
        directory = (
            Path(configured_dir).expanduser()
            if configured_dir
            else _default_log_directory()
        )
        retention_raw = os.environ.get(
            ENV_LOG_RETENTION_DAYS, str(DEFAULT_LOG_RETENTION_DAYS)
        )
        retention_days = max(1, int(retention_raw))
        debug_enabled = _parse_bool(os.environ.get(ENV_DEBUG), default=True)
        return cls(
            directory=directory,
            debug_enabled=debug_enabled,
            retention_days=retention_days,
        )

    @property
    def file_path(self) -> Path:
        """Return the active log file path."""

        return self.directory / self.file_name

    @property
    def level(self) -> int:
        """Return the active log level."""

        return logging.DEBUG if self.debug_enabled else logging.INFO

    @property
    def level_name(self) -> str:
        """Return the active log level name."""

        return logging.getLevelName(self.level)


def configure_logging(config: LoggingConfig) -> logging.Logger:
    """Configure the package logger with daily rotation."""

    config.directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()

    handler = TimedRotatingFileHandler(
        filename=str(config.file_path),
        when="D",
        interval=1,
        backupCount=config.retention_days,
        encoding="utf-8",
    )
    setattr(handler, _MANAGED_HANDLER_ATTR, True)
    handler.setLevel(config.level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.info(
        "Logging configured level=%s rotate=daily retention_days=%s file=%s",
        config.level_name,
        config.retention_days,
        config.file_path,
    )
    return logger


def render_logging_summary(config: LoggingConfig) -> str:
    """Render a compact logging summary for runtime defaults."""

    debug_mode = "on" if config.debug_enabled else "off"
    return (
        f"debug={debug_mode} level={config.level_name} "
        f"rotate=daily retention={config.retention_days}d "
        f"file={config.file_path}"
    )
