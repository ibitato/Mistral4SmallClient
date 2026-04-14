from __future__ import annotations

from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from mistral4cli.logging_config import (
    DEFAULT_LOG_RETENTION_DAYS,
    LoggingConfig,
    configure_logging,
)


def test_configure_logging_uses_daily_rotation_and_retention(tmp_path: Path) -> None:
    config = LoggingConfig(
        directory=tmp_path,
        debug_enabled=True,
        retention_days=DEFAULT_LOG_RETENTION_DAYS,
    )

    logger = configure_logging(config)
    logger.debug("debug trace")

    handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, TimedRotatingFileHandler)
    ]
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.when == "D"
    assert handler.backupCount == DEFAULT_LOG_RETENTION_DAYS
    assert Path(handler.baseFilename) == config.file_path

    rendered = config.file_path.read_text(encoding="utf-8")
    assert "INFO mistral4cli Logging configured" in rendered
    assert "DEBUG mistral4cli debug trace" in rendered


def test_configure_logging_can_disable_debug_output(tmp_path: Path) -> None:
    config = LoggingConfig(directory=tmp_path, debug_enabled=False, retention_days=2)

    logger = configure_logging(config)
    logger.debug("suppressed")
    logger.info("info trace")

    rendered = config.file_path.read_text(encoding="utf-8")
    assert "suppressed" not in rendered
    assert "INFO mistral4cli info trace" in rendered
