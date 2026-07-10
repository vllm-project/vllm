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
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None

            req_id = new_req_data.req_id
            self._remove_request(req_id)

            sampling_params = new_req_data.sampling_params
            max_tokens = (
                sampling_params.max_tokens
                if sampling_params and sampling_params.max_tokens is not None
                else 1
            )
            self._add_simulated_request_state(
                req_id=req_id,
                prompt_len=len(new_req_data.prompt_token_ids),
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=max_tokens,
            )
            self._simulated_token_ids_by_req[new_req_data.req_id] = (
                self._get_simulated_output_token_ids(new_req_data)
            )

    def update_requests(self, scheduler_output: "SchedulerOutput") -> None:
        reqs = scheduler_output.scheduled_cached_reqs
        for req_id, num_computed_tokens in zip(reqs.req_ids, reqs.num_computed_tokens):
            req_index = self.req_states.req_id_to_index[req_id]
            self.req_states.num_computed_tokens_np[req_index] = num_computed_tokens
            self.req_states.num_computed_tokens.gpu[req_index] = num_computed_tokens
            self.req_states.num_computed_prefill_tokens[req_index] = min(
                num_computed_tokens,
                self.req_states.prefill_len.np[req_index],
            )

    @staticmethod
    def _get_simulated_output_token_ids(
        new_req_data: "NewRequestData",
    ) -> list[int]:
        sampling_params = new_req_data.sampling_params
        extra_args = sampling_params.extra_args if sampling_params else None
        token_ids = (
            None if extra_args is None else extra_args.get("simulated_output_token_ids")
        )
        if token_ids is None:
            return []
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in token_ids
        ):
            raise ValueError("simulated_output_token_ids must be a list of integers.")
        return token_ids

    def _add_simulated_request_state(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        req_states = self.req_states
        assert len(req_states.free_indices) > 0, "No free indices"
        req_index = req_states.free_indices.pop()
        req_states.req_id_to_index[req_id] = req_index
        req_states.index_to_req_id[req_index] = req_id

        prefill_len = len(all_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        req_states.max_seq_len[req_index] = prompt_len + max_tokens
        req_states.prompt_len.np[req_index] = prompt_len
        req_states.prompt_len.gpu[req_index] = prompt_len
        req_states.prefill_len.np[req_index] = prefill_len
        req_states.prefill_len.gpu[req_index] = prefill_len
        req_states.total_len.gpu[req_index] = prefill_len
        req_states.num_computed_prefill_tokens[req_index] = num_computed_tokens
        req_states.num_computed_tokens_np[req_index] = num_computed_tokens
        req_states.num_computed_tokens.gpu[req_index] = num_computed_tokens
        req_states.all_token_ids.gpu[req_index, :prefill_len] = torch.tensor(
            all_token_ids,
            dtype=torch.int32,
            device=req_states.all_token_ids.gpu.device,
        )
        req_states.draft_tokens[req_index].zero_()

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
            scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_np[req_index]
            prompt_len = self.req_states.prompt_len.np[req_index]
            reaches_decode = num_computed_tokens + scheduled_tokens >= prompt_len
            self._advance_computed_tokens(req_index, scheduled_tokens)
            if not reaches_decode:
                sampled_token_ids.append([])
                continue

            sampled_token_id = self._next_simulated_token_id(req_id, req_index)
            sampled_token_ids.append([sampled_token_id])
            self._append_simulated_token(req_index, sampled_token_id)

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=sampled_token_ids,
        )

    def _advance_computed_tokens(
        self,
        req_index: int,
        scheduled_tokens: int,
    ) -> None:
        num_computed_tokens = (
            self.req_states.num_computed_tokens_np[req_index] + scheduled_tokens
        )
        self.req_states.num_computed_tokens_np[req_index] = num_computed_tokens
        self.req_states.num_computed_tokens.gpu[req_index] = num_computed_tokens
        self.req_states.num_computed_prefill_tokens[req_index] = min(
            num_computed_tokens, self.req_states.prefill_len.np[req_index]
        )

    def _append_simulated_token(
        self,
        req_index: int,
        sampled_token_id: int,
    ) -> None:
        token_idx = int(self.req_states.total_len.gpu[req_index])
        if token_idx >= self.max_model_len:
            raise ValueError(
                "Simulated sampled token IDs exceed the max model length. "
                f"Total number of tokens: {token_idx + 1} > max_model_len: "
                f"{self.max_model_len}"
            )

        self.req_states.total_len.gpu[req_index] = token_idx + 1
        self.req_states.all_token_ids.gpu[req_index, token_idx] = sampled_token_id
        self.req_states.last_sampled_tokens[req_index, 0] = sampled_token_id

    def _next_simulated_token_id(self, req_id: str, req_index: int) -> int:
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
