# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import (
    get_batch_metadata_across_dp,
    make_num_tokens_across_dp,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    post_update,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.sample.logprob import compute_prompt_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sample import rejection_sample
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]
        self.is_pooling_model = False

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()

        # Multimodal
        self.uses_mrope = self.model_config.uses_mrope
        if self.uses_mrope:
            self.mrope_states = MRopeState(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                device=self.device,
            )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.output_copy_event = torch.cuda.Event()
        if self.use_async_scheduling:
            self.input_prep_event = torch.cuda.Event()
            self.structured_outputs_event = torch.cuda.Event()
        else:
            self.input_prep_event = None
            self.structured_outputs_event = None

        if self.speculative_config is not None:
            self.do_spec_decode = True
            self.num_speculative_steps = self.speculative_config.num_speculative_tokens
            self.speculator = init_speculator(self.vllm_config, self.device)
        else:
            self.do_spec_decode = False
            self.num_speculative_steps = 0
            self.speculator = None

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            inputs_embeds_size=self.inputs_embeds_size,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            logprobs_mode=self.model_config.logprobs_mode,
        )

        # CUDA graphs.
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config, self.uses_mrope, self.device
        )
        # Structured outputs worker.
        self.structured_outputs_worker = StructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1),
            vocab_size=self.vocab_size,
        )

        # Buffers for CPU-to-GPU copies.
        self.tmp_idx_mapping = UvaBufferPool(self.max_num_reqs, torch.int32)
        self.tmp_cu_num_logits = UvaBufferPool(self.max_num_reqs + 1, torch.int32)
        self.tmp_query_start_loc = UvaBufferPool(self.max_num_reqs + 1, torch.int32)

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    def get_supported_tasks(self) -> tuple[str]:
        return ("generate",)

    def load_model(self, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.model_config,
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model,
                    self.vllm_config,
                    self.device,
                )
            if self.do_spec_decode:
                self.speculator.load_model(self.model)
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

    def get_model(self) -> nn.Module:
        return self.model

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        self.attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )
        if self.do_spec_decode:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.kv_cache_config,
                self.attn_metadata_builders,
                self.block_tables,
            )

        # TODO(woosuk): Support other backends.
        if not all(b.get_name() == "FLASH_ATTN" for b in self.attn_backends.values()):
            raise NotImplementedError("Only FLASH_ATTN backend is supported currently.")

        self.kv_caches: list[torch.Tensor] = []
        init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore

    def prepare_dummy_attn_metadata(self, input_batch: InputBatch) -> None:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        input_batch.attn_metadata = attn_metadata

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        skip_attn: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = min(num_tokens, self.max_num_reqs)
        input_batch = InputBatch.make_dummy(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            input_buffers=self.input_buffers,
            device=self.device,
        )
        if not skip_attn:
            self.prepare_dummy_attn_metadata(input_batch)

        dp_size = self.parallel_config.data_parallel_size
        num_tokens_across_dp = make_num_tokens_across_dp(dp_size, num_tokens)
        num_sampled_tokens = np.ones(input_batch.num_reqs, dtype=np.int32)
        if not self.uses_mrope:
            positions = input_batch.positions
        else:
            positions = input_batch.mrope_positions
        with (
            self.maybe_dummy_run_with_lora(
                self.lora_config,
                input_batch.num_scheduled_tokens,
                num_sampled_tokens,
            ),
            set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
            ),
        ):
            hidden_states = self.model(
                input_ids=input_batch.input_ids,
                positions=positions,
            )
            sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        self.sampler(logits, idx_mapping, idx_mapping_np, pos)

    @torch.inference_mode()
    def profile_run(self) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens,
            skip_attn=True,
        )
        self._dummy_sampler_run(sample_hidden_states)
        if self.do_spec_decode:
            num_tokens_across_dp = make_num_tokens_across_dp(
                self.parallel_config.data_parallel_size, self.max_num_tokens
            )
            self.speculator.run_model(
                self.max_num_tokens,
                attn_metadata=None,
                num_tokens_across_dp=num_tokens_across_dp,
            )
        torch.cuda.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def reset_mm_cache(self) -> None:
        pass

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        # SP is not supported yet.
        return num_scheduled_tokens

    @torch.inference_mode()
    def capture_model(self) -> int:
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        start_time = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            self.cudagraph_manager.capture(
                model=self.model,
                input_buffers=self.input_buffers,
                block_tables=self.block_tables,
                attn_metadata_builders=self.attn_metadata_builders,
                kv_cache_config=self.kv_cache_config,
            )
            if self.do_spec_decode:
                self.speculator.capture_model()

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def warmup_for_prefill(self) -> None:
        # For FlashInfer, we would like to execute a dummy prefill run
        # to trigger JIT compilation.
        if all("FLASHINFER" in b.get_name() for b in self.attn_backends.values()):
            self._dummy_run(self.max_num_tokens, skip_attn=False)
            torch.cuda.synchronize()

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        if scheduler_output.preempted_req_ids is not None:
            for req_id in scheduler_output.preempted_req_ids:
                self.req_states.remove_request(req_id)
        for req_id in scheduler_output.finished_req_ids:
            self.req_states.remove_request(req_id)

        # Add new requests.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            assert new_req_data.sampling_params is not None
            req_id = new_req_data.req_id
            prompt_len = len(new_req_data.prompt_token_ids)
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                prefill_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                sampling_params=new_req_data.sampling_params,
                lora_request=new_req_data.lora_request,
            )
            req_index = self.req_states.req_id_to_index[req_id]

            # Pre-compute M-RoPE positions for prefill.
            if self.uses_mrope:
                self.mrope_states.init_prefill_mrope_positions(
                    req_index,
                    self.model,  # type: ignore
                    new_req_data.prefill_token_ids,
                    mm_features=[],  # TODO
                )

            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.sampler.add_request(
                req_index, prompt_len, new_req_data.sampling_params
            )

        # Add new blocks for the existing requests.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            req_index = self.req_states.req_id_to_index[req_id]
            req_new_block_ids = cached_reqs.new_block_ids[i]
            if req_new_block_ids is not None:
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

        self.req_states.apply_staged_writes()
        self.block_tables.apply_staged_writes()
        self.sampler.apply_staged_writes(
            self.req_states.prefill_token_ids.gpu,
            self.req_states.prefill_len.np,
            self.req_states.prompt_len,
        )
        if self.uses_mrope:
            self.mrope_states.apply_staged_writes()

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_tokens_after_padding: int,
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(
            scheduler_output.num_scheduled_tokens.keys(),
            key=lambda k: scheduler_output.num_scheduled_tokens[k],
        )
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids], dtype=np.int32
        )

        idx_mapping_list = [
            self.req_states.req_id_to_index[req_id] for req_id in req_ids
        ]
        idx_mapping_np = np.array(idx_mapping_list, dtype=np.int32)
        idx_mapping = self.tmp_idx_mapping.copy_to_gpu(idx_mapping_np)

        # Get the number of draft tokens for each request.
        if not scheduler_output.scheduled_spec_decode_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
        else:
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens = np.array(
                [
                    len(draft_tokens[req_id]) if req_id in draft_tokens else 0
                    for req_id in req_ids
                ],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = self.tmp_cu_num_logits.copy_to_gpu(cu_num_logits_np)

            expanded_idx_mapping = expand_idx_mapping(
                idx_mapping,
                total_num_logits,
                cu_num_logits,
                max_expand_len=self.num_speculative_steps + 1,
            )

        # Block tables: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(idx_mapping)

        # Get query_start_loc.
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        self.tmp_query_start_loc.copy_to_gpu(
            query_start_loc_np,
            out=self.input_buffers.query_start_loc,
        )
        query_start_loc_np = query_start_loc_np[: num_reqs + 1]
        query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        # Get prefill tokens.
        prepare_prefill_inputs(
            self.input_buffers.input_ids,
            self.req_states.next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            self.req_states.prefill_token_ids.gpu,
            self.req_states.prefill_len.gpu,
            self.req_states.num_computed_tokens.gpu,
        )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Prepare M-RoPE positions.
        if self.uses_mrope:
            self.mrope_states.prepare_mrope_positions(
                idx_mapping,
                query_start_loc,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
                self.input_buffers.mrope_positions,
            )

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        # Compute slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc, self.input_buffers.positions[:num_tokens]
        )

        # Layer name -> attention metadata.
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self.input_buffers.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]
        mrope_positions = self.input_buffers.mrope_positions[
            :, :num_tokens_after_padding
        ]
        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            input_ids=input_ids,
            positions=positions,
            mrope_positions=mrope_positions,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        sample_pos = input_batch.positions[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        # Sample tokens and compute logprobs (if needed).
        sampler_output = self.sampler(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            sample_pos,
        )

        if input_batch.num_draft_tokens == 0:
            # No draft tokens (common case).
            num_sampled = torch.ones(
                input_batch.num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            # Rejection sampling for spec decoding.
            input_ids = input_batch.input_ids[input_batch.logits_indices]
            sampled_tokens, num_sampled = rejection_sample(
                sampler_output.sampled_token_ids,
                input_ids,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
            sampler_output.sampled_token_ids = sampled_tokens

        # Get the number of sampled and rejected tokens.
        # For chunked prefills, num_sampled and num_rejected are both 0.
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return sampler_output, num_sampled, num_rejected

    def compute_prompt_logprobs(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.req_states.needs_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # No request asks for prompt logprobs.
            return {}

        prompt_lens = self.req_states.prompt_len[idx_mapping_np]
        # NOTE(woosuk): -1 because the last prompt token's hidden state is not
        # needed for prompt logprobs.
        computed_prefill = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
        includes_prompt = computed_prefill < prompt_lens - 1
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = (
            prompt_lens < self.req_states.prefill_len.np[idx_mapping_np]
        )
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # Just to be safe, clone the input ids.
        n = input_batch.num_tokens
        # Shift the input ids by one.
        token_ids = torch.empty_like(input_batch.input_ids[:n])
        token_ids[: n - 1] = input_batch.input_ids[1:n]
        # To avoid out-of-bound access, set the last token id to 0.
        token_ids[n - 1] = 0

        # Handle chunked prompts.
        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens
        prefill_token_ids = self.req_states.prefill_token_ids.gpu
        query_start_loc_np = input_batch.query_start_loc_np
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue
            if not is_prompt_chunked[i]:
                continue
            # The prompt is chunked. Get the next prompt token.
            req_idx = input_batch.idx_mapping_np[i]
            idx = int(query_start_loc_np[i + 1] - 1)
            # NOTE(woosuk): This triggers two GPU operations.
            next_prompt_token = prefill_token_ids[req_idx, pos_after_step[i]]
            token_ids[idx] = next_prompt_token

        # NOTE(woosuk): We mask out logprobs for negative tokens.
        prompt_logprobs, prompt_ranks = compute_prompt_logprobs(
            token_ids,
            hidden_states[:n],
            self.model.compute_logits,
        )

        prompt_token_ids = token_ids.unsqueeze(-1)
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            logprobs = LogprobsTensors(
                logprob_token_ids=prompt_token_ids[start_idx:end_idx],
                logprobs=prompt_logprobs[start_idx:end_idx],
                selected_token_ranks=prompt_ranks[start_idx:end_idx],
            )

            req_extra_data = self.req_states.extra_data[req_id]
            prompt_logprobs_list = req_extra_data.in_progress_prompt_logprobs
            if is_prompt_chunked[i]:
                # Prompt is chunked. Do not return the logprobs yet.
                prompt_logprobs_list.append(logprobs)
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                prompt_logprobs_list.append(logprobs)
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.cat(
                        [x.logprob_token_ids for x in prompt_logprobs_list]
                    ),
                    logprobs=torch.cat([x.logprobs for x in prompt_logprobs_list]),
                    selected_token_ranks=torch.cat(
                        [x.selected_token_ranks for x in prompt_logprobs_list]
                    ),
                )
                prompt_logprobs_list.clear()

            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict

    def postprocess(
        self,
        input_batch: InputBatch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        # Update the number of computed tokens.
        post_update(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.sampler.penalties_state.output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        # Update the number of computed prefill tokens.
        idx_mapping_np = input_batch.idx_mapping_np
        computed_prefill = self.req_states.num_computed_prefill_tokens
        # TODO(woosuk): Simplify this.
        computed_prefill[idx_mapping_np] = np.minimum(
            computed_prefill[idx_mapping_np] + input_batch.num_scheduled_tokens,
            self.req_states.prefill_len.np[idx_mapping_np],
        )

    @torch.inference_mode()
    def propose_draft(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> torch.Tensor:
        assert self.speculator is not None
        last_sampled_tokens = self.req_states.last_sampled_tokens[
            input_batch.idx_mapping
        ]
        next_prefill_tokens = self.req_states.next_prefill_tokens[
            input_batch.idx_mapping
        ]
        draft_tokens = self.speculator.propose(
            input_batch,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled_tokens,
            next_prefill_tokens,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
        )
        return draft_tokens

    def get_cudagraph_and_dp_padding(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        dp_size = self.parallel_config.data_parallel_size
        if dp_size == 1:
            # No DP. Only consider CUDA graphs.
            if total_num_scheduled_tokens == 0:
                # Special case: no tokens to run.
                return CUDAGraphMode.NONE, 0, None

            cudagraph_size = self.cudagraph_manager.get_cudagraph_size(
                scheduler_output, total_num_scheduled_tokens
            )
            if cudagraph_size is not None:
                # Use full CUDA graph.
                return CUDAGraphMode.FULL, cudagraph_size, None
            # Fall back to eager mode.
            # TODO(woosuk): Support piecewise CUDA graphs.
            return CUDAGraphMode.NONE, total_num_scheduled_tokens, None

        # Consider DP padding and CUDA graph.
        if total_num_scheduled_tokens == 0:
            # Special handling is needed for 0.
            cudagraph_size_before_dp: int | None = 0
        else:
            cudagraph_size_before_dp = self.cudagraph_manager.get_cudagraph_size(
                scheduler_output, total_num_scheduled_tokens
            )
            if cudagraph_size_before_dp is None:
                cudagraph_size_before_dp = -1

        assert cudagraph_size_before_dp is not None
        dp_rank = self.parallel_config.data_parallel_rank
        num_tokens_across_dp, cudagraph_size_across_dp = get_batch_metadata_across_dp(
            total_num_scheduled_tokens,
            cudagraph_size_before_dp,
            dp_size,
            dp_rank,
        )
        if all(cudagraph_size_across_dp >= 0):
            # If all ranks can use CUDA graph, pad to the maximum number of tokens
            # across DP and use CUDA graph.
            num_tokens_after_padding = int(cudagraph_size_across_dp.max().item())
            cudagraph_mode = CUDAGraphMode.FULL
        else:
            # If any of the ranks cannot use CUDA graph, use eager mode for all ranks.
            # No padding is needed except for ranks that have no tokens to run.
            num_tokens_across_dp = torch.clamp(num_tokens_across_dp, min=1)
            num_tokens_after_padding = num_tokens_across_dp[dp_rank]
            cudagraph_mode = CUDAGraphMode.NONE
        return cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
    ) -> ModelRunnerOutput | None:
        assert intermediate_tensors is None
        if scheduler_output.total_num_scheduled_tokens == 0 and not dummy_run:
            # No need to run the model.
            self.update_states(scheduler_output)
            return EMPTY_MODEL_RUNNER_OUTPUT

        cudagraph_mode, num_tokens_after_padding, num_tokens_across_dp = (
            self.get_cudagraph_and_dp_padding(scheduler_output)
        )
        self.update_states(scheduler_output)
        if num_tokens_after_padding == 0:
            # All DP ranks have zero tokens to run.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(
                scheduler_output,
                num_tokens_after_padding,
            )
            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.req_states.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)
        else:
            # No actual tokens to run. A dummy run for DP.
            num_reqs = min(num_tokens_after_padding, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens_after_padding,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            self.prepare_dummy_attn_metadata(input_batch)

        # Run model.
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Run CUDA graph.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            hidden_states = self.cudagraph_manager.run(
                input_batch.num_tokens_after_padding
            )
        else:
            # Run PyTorch model in eager mode.
            # TODO(woosuk): Support piecewise CUDA graph.
            if not self.uses_mrope:
                positions = input_batch.positions
            else:
                positions = input_batch.mrope_positions
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                cudagraph_runtime_mode=cudagraph_mode,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=positions,
                )

        self.execute_model_state = hidden_states, input_batch
        return None

    @torch.inference_mode()
    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> AsyncOutput | ModelRunnerOutput:
        assert self.execute_model_state is not None
        hidden_states, input_batch = self.execute_model_state
        self.execute_model_state = None  # type: ignore

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        prompt_logprobs_dict = self.compute_prompt_logprobs(hidden_states, input_batch)

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            copy_stream=self.output_copy_stream,
            copy_event=self.output_copy_event,
        )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )
        if self.do_spec_decode:
            draft_tokens = self.propose_draft(
                input_batch,
                hidden_states,
                None,  # aux_hidden_states
                num_sampled,
                num_rejected,
            )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()
