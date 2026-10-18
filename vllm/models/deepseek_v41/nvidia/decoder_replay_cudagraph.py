# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the decoder replay layers on a step's replay batch."""

from collections.abc import Callable
from copy import copy
from dataclasses import replace
from functools import partial

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


class DecoderReplayCudaGraphManager(CudaGraphManager):
    """Breakable PIECEWISE graphs of the replay layers keyed by the replay batch's
    padded size, one per replay stage. They read the layer inputs from static
    buffers. The stages capture together, on one stream: a stream's graph pool
    blocks are not reused by another's, so separate captures would add up."""

    def __init__(self, vllm_config: VllmConfig, device, layers: DecoderReplayLayers):
        compilation = copy(vllm_config.compilation_config)
        scheduler = vllm_config.scheduler_config
        sizes = compilation.decoder_replay_cudagraph_capture_sizes
        if not sizes:
            bound = min(
                compilation.max_cudagraph_capture_size,
                scheduler.max_num_seqs * layers.window,
            )
            # Replay batches hold at least `window` rows: step by window, or by 1/8
            # of the power-of-two interval once wider (256 in [2048, 4096)).
            sizes, s = [bound], layers.window
            while s < bound:
                sizes.append(s)
                s += max(layers.window, 1 << max(s.bit_length() - 4, 0))
        sizes = sorted(set(sizes))
        if max(sizes) > scheduler.max_num_batched_tokens:
            raise ValueError(
                "decoder_replay_cudagraph_capture_sizes must not exceed "
                "max_num_batched_tokens"
            )
        compilation.cudagraph_capture_sizes = sizes
        compilation.max_cudagraph_capture_size = max(sizes)
        replay_config = copy(vllm_config)
        replay_config.compilation_config = compilation
        super().__init__(replay_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.layers = layers
        num_stages = len(layers.stage_rows)
        runners = [layers.run_layers]
        if num_stages > 1:
            assert layers.run_stage is not None
            runners = [partial(layers.run_stage, i) for i in range(num_stages)]
        self.stage_runners = [
            BreakableCUDAGraphWrapper(runner, replay_config) for runner in runners
        ]
        self.breakable_cg_runner: BreakableCUDAGraphWrapper = self.stage_runners[0]
        config, act = vllm_config.model_config.hf_config, vllm_config.model_config.dtype
        assert isinstance(act, torch.dtype)
        hc, h, f32 = config.hc_mult, config.hidden_size, torch.float32
        # DeepseekV4Model.forward casts input_ids to int64 for native MegaMoE only.
        native = vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        ids = torch.int64 if native else torch.int32
        # hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
        shapes = [(h,), (), (), (hc,), (hc, 1), (hc, hc), (hc, h)]
        dtypes = [act, torch.int64, ids, f32, f32, f32, act]
        n = compilation.max_cudagraph_capture_size
        # Stages alternate between two sets: a stage's outputs may alias its
        # inputs, and the next stage gathers its inputs from them.
        self.stage_input_buffers = [
            [torch.zeros(n, *s, dtype=d, device=device) for s, d in zip(shapes, dtypes)]
            for _ in range(min(num_stages, 2))
        ]

    def capture_replay_graphs(
        self, prepare: Callable[[BatchExecutionDescriptor], None]
    ) -> None:
        """Capture each size on the replay batch ``prepare`` sets on the layers."""

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            prepare(desc)
            replay_batch = self.layers.replay_batch
            assert replay_batch is not None
            contexts = [
                replay_batch.forward_context,
                *(stage.forward_context for stage in replay_batch.stages),
            ]

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                for i, (runner, context) in enumerate(
                    zip(self.stage_runners, contexts, strict=True)
                ):
                    buffers = self.stage_input_buffers[i % 2]
                    with override_forward_context(
                        replace(context, cudagraph_runtime_mode=cg_mode)
                    ):
                        runner(*(buf[: desc.num_tokens] for buf in buffers))

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay CUDA graphs")
        self.layers.replay_batch = None

    def run(
        self,
        rows: torch.Tensor | list[torch.Tensor],
        inputs: tuple | list,
        stage: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        """Replay stage ``stage``'s graph for the forward context on ``rows`` of
        ``inputs`` (one row index per input when given a list)."""
        per_input = rows if isinstance(rows, list) else [rows] * len(inputs)
        num_rows = per_input[0].shape[0]
        buffers = self.stage_input_buffers[stage % 2]
        for buf, t, index in zip(buffers, inputs, per_input):
            torch.index_select(t, 0, index, out=buf[:num_rows])
        batch_descriptor = get_forward_context().batch_descriptor
        assert batch_descriptor is not None
        num_tokens = batch_descriptor.num_tokens
        outputs = self.stage_runners[stage](*(buf[:num_tokens] for buf in buffers))
        return tuple(out[:num_rows] for out in outputs)
