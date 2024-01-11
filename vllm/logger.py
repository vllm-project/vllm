# Copyright © [2023,] 2023, Oracle and/or its affiliates.
"""Logging configuration for vLLM."""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from loguru import logger


class InterceptHandler(logging.Handler):
    loglevel_mapping = {
        50: 'CRITICAL',
        40: 'ERROR',
        30: 'WARNING',
        20: 'INFO',
        10: 'DEBUG',
        0: 'NOTSET',
    }

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except AttributeError:
            level = self.loglevel_mapping[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        log = logger.bind(name='app')
        log.opt(depth=depth,
                exception=record.exc_info).log(level, record.getMessage())


class CustomizeLogger:

    @classmethod
    def make_logger(cls, config_path: Path):
        config = cls.load_logging_config(config_path)
        logging_config = config.get('logger')

        logger = cls.customize_logging(
            structured_filepath=logging_config.get('structured_log_file_path'),
            unstructured_filepath=logging_config.get(
                "unstructured_log_file_path"),
            level=logging_config.get('level'),
            retention=logging_config.get('retention'),
            rotation=logging_config.get('rotation'),
            format=logging_config.get('format'),
        )

        return logger

    @classmethod
    def serialize(cls, record):
        error, exception = "", record["exception"]
        if exception:
            type, ex, tb = exception
            error = f" {type.__name__}: {ex}\n{''.join(traceback.format_tb(tb))}"

        subset = {
            "module": record["module"],
            "pathname": record["file"].name,
            "lineno": record["line"],
            "thread": record["thread"].id,
            "extra_info": record["extra"],
            "funcName": record["function"],
            "ts": int(time.time() * 1000),
            "level": record["level"].name,
            "msg": record["message"] + error,
        }
        return json.dumps(subset)

    @classmethod
    def formatter(cls, record):
        # Note this function returns the string to be formatted, not the actual message to be logged
        record["extra"]["serialized"] = cls.serialize(record)
        return "{extra[serialized]}\n"

    @classmethod
    def customize_logging(
        cls,
        structured_filepath: Path,
        unstructured_filepath: Path,
        level: str,
        rotation: str,
        retention: str,
        format: str,
    ):
        logger.remove()
        logger.add(
            sys.stdout,
            enqueue=True,
            backtrace=True,
            level=level.upper(),
            format=format,
        )
        logger.add(
            str(unstructured_filepath),
            rotation=rotation,
            retention=retention,
            enqueue=True,
            backtrace=True,
            level=level.upper(),
            serialize=False,
            format=format,
        )
        logger.add(
            str(structured_filepath),
            rotation=rotation,
            retention=retention,
            enqueue=True,
            backtrace=True,
            level=level.upper(),
            serialize=False,
            format=cls.formatter,
        )
        logging.basicConfig(handlers=[InterceptHandler()], level=0)
        logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
        for _log in ['uvicorn', 'uvicorn.error', 'fastapi']:
            _logger = logging.getLogger(_log)
            _logger.handlers = [InterceptHandler()]

        return logger.bind(name="vllm")

    @classmethod
    def load_logging_config(cls, config_path):
        config = None
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config


dir_path = os.path.dirname(os.path.realpath(__file__))
default_config_path = f"{dir_path}/logging_config.json"
_root_logger = None


def setup_logger(config_path: Optional[Path] = default_config_path):
    global _root_logger
    _root_logger = CustomizeLogger.make_logger(config_path)


def init_logger(name: str):
    if _root_logger is None:
        setup_logger()
    return _root_logger.bind(name=name)
