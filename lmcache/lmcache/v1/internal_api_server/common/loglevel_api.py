# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import logging

# Third Party
from fastapi import APIRouter
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.get("/loglevel")
async def get_or_set_log_level(
    logger_name: Optional[str] = None, level: Optional[str] = None
):
    """
    Get or set the log level for a logger.
    - No parameters: List all loggers and their levels.
    - With logger_name: Get the level of the specified logger.
    - With logger_name and level: Set the level of the specified logger.
    """
    if not logger_name and not level:
        # List all loggers and their levels
        loggers = logging.Logger.manager.loggerDict
        result = "=== Loggers and Levels ===\n"
        for name, logger_obj in loggers.items():
            if isinstance(logger_obj, logging.Logger):
                result += "%s: %s\n" % (name, logging.getLevelName(logger_obj.level))
        return PlainTextResponse(content=result, media_type="text/plain")
    elif logger_name and not level:
        # Get the level of the specified logger
        target_logger = logging.getLogger(logger_name)
        return PlainTextResponse(
            content="%s: %s" % (logger_name, logging.getLevelName(target_logger.level)),
            media_type="text/plain",
        )
    elif logger_name and level:
        # Set the level of the specified logger
        target_logger = logging.getLogger(logger_name)
        try:
            level_value = getattr(logging, level.upper())
            target_logger.setLevel(level_value)
            # Set the level of all handlers
            for handler in target_logger.handlers:
                handler.setLevel(level_value)
            return PlainTextResponse(
                content="Set %s level to %s (including all handlers)"
                % (logger_name, level.upper()),
                media_type="text/plain",
            )
        except AttributeError:
            return PlainTextResponse(
                content="Invalid log level: %s" % level,
                media_type="text/plain",
                status_code=400,
            )
