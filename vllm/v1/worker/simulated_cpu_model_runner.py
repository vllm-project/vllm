# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.cpu.model_runner import CPUModelRunner
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_states import init_model_state

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class _NoOpKVBlockZeroer:
    def zero_block_ids(self, block_ids: list[int]) -> None:
        pass


class SimulatedCPUModelRunner(CPUModelRunner):
    """CPU runner that simulates model execution while preserving scheduler/KV logic."""

    @instrument(span_name="Loading (Simulated CPU)")
    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        logger.info(
            "Initializing metadata-only model %s for simulated forward...",
            self.model_config.model,
        )
        self.compilation_config.static_forward_context.clear()
        with set_default_torch_dtype(self.model_config.dtype), torch.device("meta"):
            self.model = initialize_model(vllm_config=self.vllm_config)
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device
        )
        self.decode_query_len = (
            self.num_speculative_steps
            + self.model_state.num_new_sampled_tokens_per_step
        )
        self._simulated_token_ids_by_req: dict[str, list[int] | None] = {}
        self._total_len_by_req: dict[str, int] = {}
        logger.info(
            "Initialized metadata-only model %s on the meta device for "
            "simulated forward.",
            self.model_config.model,
        )

    def _remove_request(self, req_id: str) -> bool:
        self._simulated_token_ids_by_req.pop(req_id, None)
        self._total_len_by_req.pop(req_id, None)
        return super()._remove_request(req_id)

    def add_requests(self, scheduler_output: "SchedulerOutput") -> None:
        # Mirrors the V2 runner request bookkeeping, but skips sampler
        # registration because simulated execute_model returns tokens directly.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                max_tokens=(sampling_params.max_tokens if sampling_params else None)
                or 1,
            )
            req_index = self.req_states.req_id_to_index[req_id]
            self._total_len_by_req[req_id] = len(new_req_data.prefill_token_ids)

            if self.encoder_cache is not None:
                self.encoder_cache.add_request(req_id, new_req_data.mm_features)

            self.model_state.add_request(req_index, new_req_data)
            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)
            extra_args = sampling_params.extra_args if sampling_params else None
            token_ids = (
                None
                if extra_args is None
                else extra_args.get("simulated_output_token_ids")
            )
            self._simulated_token_ids_by_req[req_id] = self._parse_simulated_token_ids(
                token_ids
            )

        # No staged writes: simulated execution reads CPU-side mirrors only and
        # never runs sampler/model kernels that consume the GPU buffers.

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
            self._advance_request(req_id, req_index, scheduled_tokens, sampled_ids)

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=sampled_token_ids,
        )

    def _advance_request(
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

        start_idx = self._total_len_by_req[req_id]
        end_idx = start_idx + len(sampled_ids)
        if end_idx > self.max_model_len:
            raise ValueError(
                "Simulated sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

        self._total_len_by_req[req_id] = end_idx
        self.req_states.last_sampled_tokens[req_index, 0] = sampled_ids[-1]

    def _next_simulated_token_id(self, req_id: str) -> int:
        req_index = self.req_states.req_id_to_index[req_id]
        token_ids = self._simulated_token_ids_by_req[req_id]
        output_idx = int(
            self._total_len_by_req[req_id] - self.req_states.prompt_len.np[req_index]
        )
        if isinstance(token_ids, list) and output_idx < len(token_ids):
            return int(token_ids[output_idx])
        return 0

    @staticmethod
    def _parse_simulated_token_ids(
        token_ids: list[int | str] | str | None,
    ) -> list[int] | None:
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
