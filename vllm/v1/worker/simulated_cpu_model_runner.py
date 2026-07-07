# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.tracing import instrument
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class SimulatedCPUModelRunner(CPUModelRunner):
    """CPU runner that simulates model execution while preserving scheduler/KV logic."""

    @instrument(span_name="Loading (Simulated CPU)")
    def load_model(self, load_dummy_weights: bool = False) -> None:
        logger.info(
            "Initializing metadata-only model %s for simulated forward...",
            self.model_config.model,
        )
        self.compilation_config.static_forward_context.clear()
        with set_default_torch_dtype(self.model_config.dtype), torch.device("meta"):
            self.model = initialize_model(vllm_config=self.vllm_config)
        logger.info(
            "Initialized metadata-only model %s on the meta device for "
            "simulated forward.",
            self.model_config.model,
        )

    @instrument(span_name="Warmup (Simulated CPU)")
    def warming_up_model(self) -> None:
        logger.info("Skipping model warmup for simulated forward.")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Any | None = None,
    ) -> Any:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        deferred_state_corrections_fn = self._update_states(scheduler_output)
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()
        sampled_token_ids: list[list[int]] = []

        for req_idx, req_id in enumerate(req_ids_output_copy):
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            req_state = self.requests[req_id]
            reaches_decode = (
                req_state.num_computed_tokens + scheduled_tokens
                >= req_state.num_prompt_tokens
            )
            sampled_ids = (
                [self._next_simulated_token_id(req_id)] if reaches_decode else []
            )
            sampled_token_ids.append(sampled_ids)

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            if end_idx > self.max_model_len:
                raise ValueError(
                    "Simulated sampled token IDs exceed the max model length. "
                    f"Total number of tokens: {end_idx} > max_model_len: "
                    f"{self.max_model_len}"
                )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            req_state.output_token_ids.extend(sampled_ids)

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=sampled_token_ids,
        )

    def _next_simulated_token_id(self, req_id: str) -> int:
        req_state = self.requests[req_id]
        sampling_params = req_state.sampling_params
        extra_args = sampling_params.extra_args if sampling_params else None
        token_ids = (
            None if extra_args is None else extra_args.get("simulated_output_token_ids")
        )
        token_ids = self._parse_simulated_token_ids(token_ids)
        output_idx = len(req_state.output_token_ids)
        if isinstance(token_ids, list) and output_idx < len(token_ids):
            return int(token_ids[output_idx])
        return 0

    @staticmethod
    def _parse_simulated_token_ids(token_ids: Any) -> list[int] | None:
        if isinstance(token_ids, list):
            return [int(token_id) for token_id in token_ids]
        if isinstance(token_ids, str):
            try:
                parsed = json.loads(token_ids)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in token_ids.split(",")]
            if isinstance(parsed, list):
                return [int(token_id) for token_id in parsed]
        return None

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self._mamba_bufs = None
        self.attn_groups = []
        kernel_block_sizes = self._virtual_kernel_block_sizes(kv_cache_config)
        self._kernel_block_sizes = kernel_block_sizes
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        self.kv_caches = []
        logger.info(
            "Initialized virtual KV cache with %d groups and %d blocks; "
            "skipped attention metadata builders and KV tensor allocation.",
            len(kv_cache_config.kv_cache_groups),
            kv_cache_config.num_blocks,
        )

    @staticmethod
    def _virtual_kernel_block_sizes(kv_cache_config: KVCacheConfig) -> list[int]:
        kernel_block_sizes = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            kernel_block_sizes.append(kv_cache_spec.block_size)
        return kernel_block_sizes

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        return
