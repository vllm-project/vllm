# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill support for last-layer FFN row pruning."""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.last_layer_ffn import current_last_layer_ffn_indices

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def last_layer_ffn_graph_key(*args: Any, **kwargs: Any) -> int:
    del args, kwargs
    indices = current_last_layer_ffn_indices()
    return 0 if indices is None else indices.numel()


class LastLayerFFNGraphWrapper(BreakableCUDAGraphWrapper):
    def _capture(self, entry, args, kwargs):
        caller = torch.cuda.current_stream()
        if not hasattr(self, "ffn_capture_stream"):
            self.ffn_capture_stream = torch.cuda.Stream(device=caller.device)
        with graph_capture(caller.device, self.ffn_capture_stream):
            super()._capture(entry, args, kwargs)
        caller.wait_stream(self.ffn_capture_stream)
        return self._replay(entry, args, kwargs)


class LastLayerFFNPruner:
    def __init__(self):
        self.enabled = envs.VLLM_ENABLE_LAST_LAYER_FFN_PRUNING
        self.indices: torch.Tensor | None = None
        self.batches = 0
        self.skipped_rows = 0

    def prepare(self, runner: "GPUModelRunner", batch, descriptor, dummy, ubatch):
        if (
            not self.enabled
            or dummy
            or not runner.is_last_pp_rank
            or ubatch is not None
            or torch.version.cuda is None
        ):
            return None
        config = runner.vllm_config
        parallel = config.parallel_config
        if (
            not getattr(runner.model, "supports_last_layer_ffn_pruning", False)
            or runner.is_pooling_model
            or runner.use_aux_hidden_state_outputs
            or config.model_config.is_multimodal_model
            or config.model_config.quantization is not None
            or config.quant_config is not None
            or config.aux_output_config.enabled
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or parallel.data_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_ubatching
            or batch.pcp is not None
            or batch.fast_prefill is not None
            or config.compilation_config.mode != CompilationMode.NONE
            or descriptor.cg_mode == CUDAGraphMode.FULL
            or (
                descriptor.cg_mode == CUDAGraphMode.PIECEWISE
                and (
                    runner.cudagraph_manager is None
                    or not runner.cudagraph_manager.use_breakable_cg
                )
            )
            or batch.num_reqs > 16
            or batch.num_tokens <= batch.num_reqs
            or batch.num_draft_tokens != 0
            or not batch.has_prefill
            or batch.logits_indices.numel() != batch.num_reqs
        ):
            return None
        prompt = runner.prompt_logprobs_worker
        if prompt is not None and (
            np.any(prompt.uses_prompt_logprobs[batch.idx_mapping_np])
            or any(req_id in prompt.token_id_scores for req_id in batch.req_ids)
        ):
            return None
        if self.indices is None:
            self.indices = torch.empty(16, dtype=torch.int64, device=runner.device)
        indices = self.indices[: batch.num_reqs]
        indices.copy_(batch.logits_indices)
        if self.batches == 0:
            logger.info("Last-layer FFN pruning active on prefills")
        self.batches += 1
        self.skipped_rows += batch.num_tokens - batch.num_reqs
        return indices
