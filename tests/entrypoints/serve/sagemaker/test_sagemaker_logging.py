# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import logging
import sys

from fastapi import FastAPI


def test_sagemaker_import_does_not_change_handler_level():
    """
    Regression test for issue #13634.

    Importing SageMaker integration should not change existing
    logging handler levels.
    """

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]

    try:
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

        sys.modules.pop(
            "vllm.entrypoints.serve.sagemaker.api_router",
            None,
        )

        api_router = importlib.import_module(
            "vllm.entrypoints.serve.sagemaker.api_router"
        )

        app = FastAPI()

        api_router.attach_router(app, ())

        assert handler.level == logging.INFO

    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(old_handlers)
