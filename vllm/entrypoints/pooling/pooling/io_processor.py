# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import Any

from vllm import PoolingParams, PoolingRequestOutput
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.renderers.inputs.preprocess import parse_model_prompt, prompt_to_seq

from ..base.io_processor import PoolingIOProcessor
from ..typing import OfflineInputsContext, OfflineOutputsContext, PoolingServeContext
from .protocol import IOProcessorRequest, IOProcessorResponse

logger = init_logger(__name__)


class PluginWithoutIOProcessorPlugins(PoolingIOProcessor):
    # Some models, such as Terratorch (tests/models/test_terratorch.py),
    # use plugin tasks in the pooler but do not use IO Processor plugins.
    name = "plugin"


class PluginWithIOProcessorPlugins(PoolingIOProcessor):
    """IO Processor plugins are a feature that allows pre- and post-processing
    of the model input and output for pooling models."""

    name = "plugin"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        assert io_processor is not None
        self.io_processor = io_processor

    #######################################
    # online APIs

    def pre_process_online(self, ctx: PoolingServeContext):
        assert isinstance(ctx.request, IOProcessorRequest)

        validated_prompt = self.io_processor.parse_data(ctx.request.data)

        raw_prompts = self.io_processor.pre_process(
            prompt=validated_prompt, request_id=ctx.request_id
        )

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(self.model_config, prompt)
            )
            for prompt in prompt_to_seq(raw_prompts)
        ]

        tok_params = ctx.request.build_tok_params(self.model_config)

        ctx.engine_inputs = self.renderer.render_cmpl(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(ctx.request, k, None)) is not None
            },
        )

        pooling_params = self.io_processor.merge_pooling_params()
        if pooling_params.task is None:
            pooling_params.task = "plugin"
        ctx.pooling_params = pooling_params

    def post_process_online(
        self,
        ctx: PoolingServeContext,
    ):
        output = self.io_processor.post_process(
            ctx.final_res_batch,
            request_id=ctx.request_id,
        )

        if callable(
            output_to_response := getattr(self.io_processor, "output_to_response", None)
        ):
            logger.warning_once(
                "`IOProcessor.output_to_response` is deprecated. To ensure "
                "consistency between offline and online APIs, "
                "`IOProcessorResponse` will become a transparent wrapper "
                "around output data from v0.19 onwards.",
            )

            if hasattr(output, "request_id") and output.request_id is None:
                output.request_id = ctx.request_id  # type: ignore

            ctx.response = output_to_response(output)  # type: ignore
        else:
            ctx.response = IOProcessorResponse(request_id=ctx.request_id, data=output)

    #######################################
    # offline APIs

    def pre_process_offline(self, ctx: OfflineInputsContext) -> Sequence[EngineInput]:
        assert isinstance(ctx.prompts, dict) and "data" in ctx.prompts
        assert ctx.pooling_params is not None

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
