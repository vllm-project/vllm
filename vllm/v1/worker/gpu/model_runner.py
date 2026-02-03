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
from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import (
    get_cudagraph_and_dp_padding,
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
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sample import rejection_sample
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
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
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        if self.supports_mm_inputs:
            self.encoder_runner = EncoderRunner(
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                dtype=self.dtype,
                device=self.device,
            )
        self.uses_mrope = self.model_config.uses_mrope
        if self.uses_mrope:
            self.mrope_states = MRopeState(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                device=self.device,
            )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.output_copy_event = torch.cuda.Event()

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
            device=self.device,
        )
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            logprobs_mode=self.model_config.logprobs_mode,
        )
        self.prompt_logprobs_worker = PromptLogprobsWorker(self.max_num_reqs)

        # CUDA graphs.
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config, self.uses_mrope, self.device
        )
        # Structured outputs worker.
        self.structured_outputs_worker = StructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1),
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    @staticmethod
    def get_supported_tasks() -> tuple[str]:
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
                    self.model, self.vllm_config, self.device
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

        prepare_communication_buffer_for_model(self.model)
        if self.do_spec_decode:
            speculator_model = getattr(self.speculator, "model", None)
            if speculator_model is not None:
                prepare_communication_buffer_for_model(speculator_model)

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
            self.kv_cache_config, self.vllm_config, self.device
        )
        if self.do_spec_decode:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.kv_cache_config,
                self.attn_metadata_builders,
                self.block_tables,
            )

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore

    def prepare_dummy_attn_metadata(self, input_batch: InputBatch) -> None:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
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
        input_batch.slot_mappings = slot_mappings_by_layer

    @torch.inference_mode()
    def _dummy_run(
        self, num_tokens: int, *args, skip_attn: bool = True, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        num_tokens_per_request = [num_tokens // num_reqs] * num_reqs
        num_tokens_per_request[-1] += num_tokens % num_reqs
        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Execute the model.
        self.execute_model(
            dummy_scheduler_output, dummy_run=True, skip_attn_for_dummy_run=skip_attn
        )
        self.kv_connector.set_disabled(False)
        assert self.execute_model_state is not None
        hidden_states, input_batch, _ = self.execute_model_state
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        self.sampler(logits, idx_mapping, idx_mapping_np, idx_mapping_np, pos)

    @torch.inference_mode()
    def profile_run(self) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True
        )
        self._dummy_sampler_run(sample_hidden_states)
        if self.do_spec_decode:
            num_tokens_across_dp = make_num_tokens_across_dp(
                self.parallel_config.data_parallel_size, self.max_num_tokens
            )
            self.speculator.run_model(
                self.max_num_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
            )
        torch.cuda.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def reset_mm_cache(self) -> None:
        if self.supports_mm_inputs:
            self.encoder_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.supports_mm_inputs:
            self.encoder_runner.reset_encoder_cache()

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
            mrope_positions = None
            if self.uses_mrope:
                mrope_positions = self.mrope_states.mrope_positions
            inputs_embeds = None
            if self.supports_mm_inputs:
                inputs_embeds = self.encoder_runner.inputs_embeds
            self.cudagraph_manager.capture(
                model=self.model,
                input_buffers=self.input_buffers,
                mrope_positions=mrope_positions,
                inputs_embeds=inputs_embeds,
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

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            self.req_states.remove_request(req_id)
            if self.supports_mm_inputs:
                self.encoder_runner.remove_request(req_id)
            self.prompt_logprobs_worker.remove_request(req_id)
            self.lora_state.remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.supports_mm_inputs:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_runner.free_encoder_cache(mm_hash)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
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
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.supports_mm_inputs:
                self.encoder_runner.add_request(req_id, new_req_data.mm_features)

            # Pre-compute M-RoPE positions for prefill.
            if self.uses_mrope:
                self.mrope_states.init_prefill_mrope_positions(
                    req_index,
                    self.model,  # type: ignore
                    new_req_data.prefill_token_ids,
                    mm_features=new_req_data.mm_features,
                )

            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.sampler.add_request(
                req_index, prompt_len, new_req_data.sampling_params
            )
            self.prompt_logprobs_worker.add_request(
                req_id, req_index, new_req_data.sampling_params
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.sampler.apply_staged_writes(
                self.req_states.prefill_token_ids.gpu,
                self.req_states.prefill_len.np,
                self.req_states.prompt_len,
            )
            if self.uses_mrope:
                self.mrope_states.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        for req_new_block_ids, req_id in zip(reqs.new_block_ids, reqs.req_ids):
            if req_new_block_ids is not None:
                req_index = self.req_states.req_id_to_index[req_id]
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_tokens_after_padding: int
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore[arg-type]
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
        else:
            num_draft_tokens = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
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
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

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
            idx_mapping,
            query_start_loc,
            self.input_buffers.positions[:num_tokens],
        )
        # Layer name -> slot mapping.
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
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
        mrope_positions = None
        if self.uses_mrope:
            mrope_positions = self.mrope_states.mrope_positions
            mrope_positions = mrope_positions[:, :num_tokens_after_padding]
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
            inputs_embeds=None,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings_by_layer,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
        )

    @torch.inference_mode()
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        self.encoder_runner.execute_mm_encoder(self.model, mm_hashes, mm_kwargs)
        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            self.req_states.prefill_len.np[input_batch.idx_mapping_np],
            self.req_states.num_computed_prefill_tokens[input_batch.idx_mapping_np],
        )
        return mm_embeds, is_mm_embed

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
            input_batch.cu_num_logits_np,
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
        computed_prefill[idx_mapping_np] += input_batch.num_scheduled_tokens
        np.minimum(
            computed_prefill, self.req_states.prefill_len.np, out=computed_prefill
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
        draft_tokens = self.speculator.propose(
            input_batch,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            self.req_states.last_sampled_tokens,
            self.req_states.next_prefill_tokens,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
        )
        return draft_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | None:
        assert intermediate_tensors is None
        if not dummy_run:
            # Update the request states.
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()
            if scheduler_output.total_num_scheduled_tokens == 0:
                # No need to run the model.
                empty_output = self.kv_connector.no_forward(scheduler_output)
                return empty_output

        # Get the CUDA graph size. None means no CUDA graph is used.
        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(
            scheduler_output.total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens.values(),
        )
        use_cudagraph, num_tokens_after_padding, num_tokens_across_dp = (
            get_cudagraph_and_dp_padding(
                scheduler_output.total_num_scheduled_tokens,
                cudagraph_size,
                self.parallel_config.data_parallel_size,
                self.parallel_config.data_parallel_rank,
            )
        )
        if num_tokens_after_padding == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        if not dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(
                scheduler_output, num_tokens_after_padding
            )
            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)

            if self.supports_mm_inputs:
                # Execute the multimodal encoder.
                mm_embeds, is_mm_embed = self.get_mm_embeddings(
                    scheduler_output.scheduled_encoder_inputs, input_batch
                )
                inputs_embeds = self.encoder_runner.get_inputs_embeds(
                    self.model, input_batch.input_ids, mm_embeds, is_mm_embed
                )
                input_batch.inputs_embeds = inputs_embeds[
                    : input_batch.num_tokens_after_padding
                ]
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            num_reqs = min(num_tokens_after_padding, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens_after_padding,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            if self.uses_mrope:
                input_batch.mrope_positions = self.mrope_states.mrope_positions[
                    :, :num_tokens_after_padding
                ]
            if not skip_attn_for_dummy_run:
                self.prepare_dummy_attn_metadata(input_batch)
            # FIXME(woosuk): Fix warmup for LoRA.

        # Run model.
        if use_cudagraph:
            # Run CUDA graph.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            self.kv_connector.pre_forward(scheduler_output)
            hidden_states = self.cudagraph_manager.run(
                input_batch.num_tokens_after_padding
            )
        else:
            # Run PyTorch model in eager mode.
            positions = input_batch.positions
            if self.uses_mrope:
                assert input_batch.mrope_positions is not None
                positions = input_batch.mrope_positions
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                # TODO(woosuk): Support piecewise CUDA graph.
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=input_batch.slot_mappings,
            ):
                self.kv_connector.pre_forward(scheduler_output)
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=positions,
                    inputs_embeds=input_batch.inputs_embeds,
                )

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)
        self.execute_model_state = hidden_states, input_batch, kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput:
        assert self.execute_model_state is not None
        hidden_states, input_batch, kv_connector_output = self.execute_model_state
        self.execute_model_state = None  # type: ignore

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.prefill_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            kv_connector_output=kv_connector_output,
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
            self.draft_tokens_handler.set_draft_tokens(input_batch, draft_tokens)

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()
