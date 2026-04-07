# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from vllm import PoolingParams, PoolingRequestOutput
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.typing import (
    OfflineInputsContext,
    OfflineOutputsContext,
)
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.renderers.inputs.preprocess import prompt_to_seq

logger = init_logger(__name__)


class PluginIOProcessor(PoolingIOProcessor):
    name = "plugin"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        # Validate the request data is valid for the loaded plugin
        prompt_data = ctx.prompts.get("data")
        if prompt_data is None:
            raise ValueError(
                "The 'data' field of the prompt is expected to contain "
                "the prompt data and it cannot be None. "
                "Refer to the documentation of the IOProcessor "
                "in use for more details."
            )
        validated_prompt = self.io_processor.parse_data(prompt_data)

        # obtain the actual model prompts from the pre-processor
        prompts = self.io_processor.pre_process(prompt=validated_prompt)
        prompts_seq = prompt_to_seq(prompts)

        params_seq: list[PoolingParams] = [
            self.io_processor.merge_pooling_params(param)
            for param in self._params_to_seq(
                ctx.pooling_params,
                len(prompts_seq),
            )
        ]
        for p in params_seq:
            if p.task is None:
                p.task = "plugin"

        ctx.pooling_params = params_seq
        ctx.prompts = prompts_seq
        return super().pre_process_offline(ctx)

    def post_process_offline(
        self,
        ctx: OfflineOutputsContext,
    ) -> list[PoolingRequestOutput]:
        processed_outputs = self.io_processor.post_process(ctx.outputs)

        return [
            PoolingRequestOutput[Any](
                request_id="",
                outputs=processed_outputs,
                num_cached_tokens=getattr(processed_outputs, "num_cached_tokens", 0),
                prompt_token_ids=[],
                finished=True,
            )
        ]
