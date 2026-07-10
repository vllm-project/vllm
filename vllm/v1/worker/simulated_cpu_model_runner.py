# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.cpu.model_runner import CPUModelRunner
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


class _NoOpKVBlockZeroer:
    def zero_block_ids(self, block_ids: list[int]) -> None:
        pass


class SimulatedCPUModelRunner(CPUModelRunner):
    """CPU runner that simulates model execution while preserving scheduler/KV logic."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._simulated_token_ids_by_req: dict[str, list[int]] = {}

    def _remove_request(self, req_id: str) -> bool:
        self._simulated_token_ids_by_req.pop(req_id, None)
        return super()._remove_request(req_id)

    def add_requests(self, scheduler_output: "SchedulerOutput") -> None:
        super().add_requests(scheduler_output)

        for new_req_data in scheduler_output.scheduled_new_reqs:
            self._simulated_token_ids_by_req[new_req_data.req_id] = (
                self._parse_simulated_output(new_req_data)
            )

    @staticmethod
    def _parse_simulated_output(new_req_data: "NewRequestData") -> list[int]:
        sampling_params = new_req_data.sampling_params
        extra_args = sampling_params.extra_args if sampling_params else None
        token_ids: str | None = None
        if extra_args is not None:
            raw_token_ids = extra_args.get("simulated_output_token_ids")
            if raw_token_ids is not None and not isinstance(raw_token_ids, str):
                raise ValueError(
                    "simulated_output_token_ids must be a comma-separated string."
                )
            token_ids = raw_token_ids
        if token_ids is None:
            return []
        return [int(token_id) for token_id in token_ids.split(",") if token_id.strip()]

    @instrument(span_name="Warmup (Simulated CPU)")
    def warming_up_model(self) -> None:
        logger.info("Skipping model warmup for simulated forward.")

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> ModelRunnerOutput:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if dummy_run:
            # Simulated forward has no profiling/warmup kernels to run, but
            # keeps the MRV2 execute_model signature.
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.update_pp_decode_requests()
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)

        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        req_ids_output_copy = list(scheduler_output.num_scheduled_tokens)
        req_id_to_index_output_copy = {
            req_id: i for i, req_id in enumerate(req_ids_output_copy)
        }
        sampled_token_ids: list[list[int]] = []

        for req_id in req_ids_output_copy:
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_np[req_index]
            prompt_len = self.req_states.prompt_len.np[req_index]
            reaches_decode = num_computed_tokens + scheduled_tokens >= prompt_len
            sampled_ids = (
                [self._next_simulated_token_id(req_id)] if reaches_decode else []
            )
            sampled_token_ids.append(sampled_ids)
            self._advance_simulated_request(
                req_id, req_index, scheduled_tokens, sampled_ids
            )

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=sampled_token_ids,
        )

    def _advance_simulated_request(
        self,
        req_id: str,
        req_index: int,
        scheduled_tokens: int,
        sampled_ids: list[int],
    ) -> None:
        num_computed_tokens = (
            self.req_states.num_computed_tokens_np[req_index] + scheduled_tokens
        )
        self.req_states.num_computed_tokens_np[req_index] = num_computed_tokens
        self.req_states.num_computed_prefill_tokens[req_index] = min(
            num_computed_tokens, self.req_states.prefill_len.np[req_index]
        )

        if not sampled_ids:
            return

        start_idx = int(self.req_states.total_len.gpu[req_index])
        end_idx = start_idx + len(sampled_ids)
        if end_idx > self.max_model_len:
            raise ValueError(
                "Simulated sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

        self.req_states.total_len.gpu[req_index] = end_idx
        self.req_states.last_sampled_tokens[req_index, 0] = sampled_ids[-1]

    def _next_simulated_token_id(self, req_id: str) -> int:
        req_index = self.req_states.req_id_to_index[req_id]
        token_ids = self._simulated_token_ids_by_req[req_id]
        output_idx = int(
            self.req_states.total_len.gpu[req_index]
            - self.req_states.prompt_len.np[req_index]
        )
        if output_idx < len(token_ids):
            return token_ids[output_idx]
        return 0

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Initialize V2 KV-cache metadata and block tables, but skip allocating
        # the backing KV tensors because simulated forward never runs attention.
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        block_table_max_model_len = self.max_model_len
        if self.is_encoder_decoder:
            block_table_max_model_len = max(
                block_table_max_model_len,
                self.scheduler_config.max_num_encoder_input_tokens,
                getattr(self.model_config.hf_config, "max_source_positions", 0),
            )

        block_sizes: list[int] = []
        max_num_blocks_per_group: list[int] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
            # Align to a multiple of (128 / block_size) as required by some
            # attention backends such as TRTLLM (#39324).
            if spec.block_size <= 128:
                alignment = 128 // spec.block_size
                max_num_blocks = cdiv(max_num_blocks, alignment) * alignment
            if isinstance(spec, MambaSpec):
                max_num_blocks = (
                    max_num_blocks if self.cache_config.enable_prefix_caching else 1
                ) + spec.num_speculative_blocks
            max_num_blocks_per_group.append(max_num_blocks)

        self.attn_groups, _, self.kernel_block_sizes = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=max_num_blocks_per_group,
            device=self.device,
            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
            cp_rank=self.dcp_rank,
            cp_interleave=self.cp_interleave,
        )
        initialize_mamba_ssu_backend(
            self.vllm_config.mamba_config, self.kv_cache_config
        )
        self.kv_caches = []
        self.kv_block_zeroer = _NoOpKVBlockZeroer()  # type: ignore[assignment]
        logger.info(
            "Initialized virtual KV cache with %d groups and %d blocks; "
            "skipped KV tensor allocation.",
            len(kv_cache_config.kv_cache_groups),
            kv_cache_config.num_blocks,
        )
