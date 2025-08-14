# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Any, Dict

from vllm.config import VllmConfig
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

logger = init_logger(__name__)


class AsyncMultiModalPreprocessor:

    def __init__(self, vllm_config: VllmConfig):
        logger.info("Initializing AsyncMultiModalPreprocessor with THREADS...")

        # Initialize all dependencies ONCE in the main process,
        # after that other threads can use this without re-initialization.
        tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            lora_config=vllm_config.lora_config,
        )

        self.preprocessor = InputPreprocessor(
            vllm_config.model_config,
            tokenizer,
            MULTIMODAL_REGISTRY,
        )
        logger.info(
            "Preprocessor initialized and ready for threaded execution.")

    async def preprocess(self, prompt: PromptType) -> Dict[str, Any]:
        # asyncio.to_thread to spawn a new thread
        # for preprocess to unblock the process faster than mp
        return await asyncio.to_thread(self.preprocessor.preprocess, prompt)
