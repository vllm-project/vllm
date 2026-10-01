# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graphs of the decoder replay layers on a trimming step's replay batch."""

from collections.abc import Callable
from dataclasses import replace

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


class DecoderReplayCudaGraphManager(CudaGraphManager):
    """Breakable PIECEWISE graphs of the replay layers keyed by the replay batch's
    padded size. They read the layer inputs from static buffers."""

    def __init__(self, vllm_config: VllmConfig, device, layers: DecoderReplayLayers):
        super().__init__(vllm_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.layers = layers
        self.breakable_cg_runner: BreakableCUDAGraphWrapper = BreakableCUDAGraphWrapper(
            layers.run_layers, vllm_config
        )
        config, act = vllm_config.model_config.hf_config, vllm_config.model_config.dtype
        assert isinstance(act, torch.dtype)
        hc, h, f32 = config.hc_mult, config.hidden_size, torch.float32
        mega_moe = vllm_config.kernel_config.moe_backend in MEGA_MOE_BACKENDS
        ids = torch.int64 if mega_moe else torch.int32
        # hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual
        shapes = [(h,), (), (), (hc,), (hc, 1), (hc, hc), (hc, h)]
        dtypes = [act, torch.int64, ids, f32, f32, f32, act]
        n = self.compilation_config.max_cudagraph_capture_size
        self.inputs = [
            torch.zeros(n, *s, dtype=d, device=device) for s, d in zip(shapes, dtypes)
        ]

    def capture_replay_graphs(self, prepare: Callable[[Desc], None]) -> None:
        """Capture each size on the replay batch ``prepare`` sets on the layers."""

        def create_forward_fn(desc: Desc, warmup: bool):
            prepare(desc)
            assert self.layers.replay_batch is not None
            context = self.layers.replay_batch.forward_context
            inputs = [buf[: desc.num_tokens] for buf in self.inputs]

            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                with override_forward_context(
                    replace(context, cudagraph_runtime_mode=cg_mode)
                ):
                    self.breakable_cg_runner(*inputs)

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay CUDA graphs")
        self.layers.replay_batch = None

    def run(self, rows: torch.Tensor, inputs: tuple) -> tuple[torch.Tensor, ...]:
        """Replay the forward context's graph on ``rows`` of ``inputs``."""
        num_rows = rows.shape[0]
        for buf, t in zip(self.inputs, inputs):
            torch.index_select(t, 0, rows, out=buf[:num_rows])
        batch_descriptor = get_forward_context().batch_descriptor
        assert batch_descriptor is not None
        num_tokens = batch_descriptor.num_tokens
        outputs = self.breakable_cg_runner(*(buf[:num_tokens] for buf in self.inputs))
        return tuple(out[:num_rows] for out in outputs)
