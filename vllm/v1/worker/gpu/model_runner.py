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
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.sample.sampler import SamplerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput, async_barrier
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_last_token_ids,
    prepare_inputs,
)
from vllm.v1.worker.gpu.sampler import Sampler, compute_prompt_logprobs
from vllm.v1.worker.gpu.states import RequestState, SamplingMetadata
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
        self.pin_memory = is_pin_memory_available()
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
        self.hidden_size = self.model_config.get_hidden_size()

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.input_prep_event = torch.cuda.Event()

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            vocab_size=self.vocab_size,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        # CUDA graphs.
        self.cudagraph_manager = CudaGraphManager(
            vllm_config=self.vllm_config,
            device=self.device,
        )

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
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )

    def get_model(self) -> nn.Module:
        return self.model

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config, self.kv_cache_dtype)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = kv_cache_config.block_sizes

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )

        self.kv_caches: list[torch.Tensor] = []
        init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        *args,
        input_batch: InputBatch | None = None,
        skip_attn: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_batch is None:
            num_reqs = min(num_tokens, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            if not skip_attn:
                block_tables = self.block_tables.gather_block_tables(
                    input_batch.idx_mapping
                )
                slot_mappings = self.block_tables.compute_slot_mappings(
                    input_batch.query_start_loc,
                    input_batch.positions,
                )
                attn_metadata = build_attn_metadata(
                    attn_metadata_builders=self.attn_metadata_builders,
                    num_reqs=num_reqs,
                    num_tokens=num_tokens,
                    query_start_loc=self.input_buffers.query_start_loc,
                    seq_lens=self.input_buffers.seq_lens,
                    num_computed_tokens_cpu=None,
                    block_tables=block_tables,
                    slot_mappings=slot_mappings,
                    kv_cache_config=self.kv_cache_config,
                )
                input_batch.attn_metadata = attn_metadata

        with self.maybe_dummy_run_with_lora(
            self.lora_config, input_batch.num_scheduled_tokens
        ):
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
            ):
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=input_batch.positions,
                )
            sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> None:
        num_reqs = hidden_states.shape[0]
        sampling_metadata = SamplingMetadata.make_dummy(
            num_reqs=num_reqs,
            device=self.device,
        )
        logits = self.model.compute_logits(hidden_states)
        self.sampler.sample(logits, sampling_metadata)

    @torch.inference_mode()
    def profile_run(self) -> None:
        input_batch = InputBatch.make_dummy(
            num_reqs=self.max_num_reqs,
            num_tokens=self.max_num_tokens,
            input_buffers=self.input_buffers,
            device=self.device,
        )
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens,
            input_batch=input_batch,
            skip_attn=True,
        )
        self._dummy_sampler_run(sample_hidden_states)
        torch.cuda.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def reset_mm_cache(self) -> None:
        pass

    @torch.inference_mode()
    def capture_model(self) -> int:
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            self.cudagraph_manager.capture(
                model=self.model,
                input_buffers=self.input_buffers,
                block_tables=self.block_tables,
                attn_metadata_builders=self.attn_metadata_builders,
                kv_cache_config=self.kv_cache_config,
            )

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
        # For FlashInfer, we would like to execute a dummy prefill run to trigger JIT compilation.
        if all("FLASHINFER" in b.get_name() for b in self.attn_backends.values()):
            self._dummy_run(self.max_num_tokens, skip_attn=False)
            torch.cuda.synchronize()

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.preempted_req_ids:
            self.req_states.remove_request(req_id)
        for req_id in scheduler_output.finished_req_ids:
            self.req_states.remove_request(req_id)

        # TODO(woosuk): Change SchedulerOutput.
        req_indices: list[int] = []
        cu_num_new_blocks = tuple(
            [0] for _ in range(self.block_tables.num_kv_cache_groups)
        )
        new_block_ids = tuple([] for _ in range(self.block_tables.num_kv_cache_groups))
        overwrite: list[bool] = []

        # Add new requests.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=len(new_req_data.prompt_token_ids),
                prefill_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                sampling_params=new_req_data.sampling_params,
                lora_request=new_req_data.lora_request,
            )

            req_index = self.req_states.req_id_to_index[req_id]
            req_indices.append(req_index)
            for i, block_ids in enumerate(new_req_data.block_ids):
                x = cu_num_new_blocks[i][-1]
                cu_num_new_blocks[i].append(x + len(block_ids))
                new_block_ids[i].extend(block_ids)
            overwrite.append(True)

        # Add new blocks for the existing requests.
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            req_index = self.req_states.req_id_to_index[req_id]

            req_new_block_ids = cached_reqs.new_block_ids[i]
            if req_new_block_ids is not None:
                req_indices.append(req_index)
                for group_id, block_ids in enumerate(req_new_block_ids):
                    x = cu_num_new_blocks[group_id][-1]
                    cu_num_new_blocks[group_id].append(x + len(block_ids))
                    new_block_ids[group_id].extend(block_ids)
                overwrite.append(False)

        if req_indices:
            self.block_tables.append_block_ids(
                req_indices=req_indices,
                cu_num_new_blocks=cu_num_new_blocks,
                new_block_ids=new_block_ids,
                overwrite=overwrite,
            )

    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        use_cudagraph: bool,
        padded_num_tokens: int | None,
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(
            scheduler_output.num_scheduled_tokens,
            key=scheduler_output.num_scheduled_tokens.get,
        )
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids], dtype=np.int32
        )
        if use_cudagraph:
            assert padded_num_tokens is not None
            num_tokens_after_padding = padded_num_tokens
        else:
            num_tokens_after_padding = num_tokens

        idx_mapping_list = [
            self.req_states.req_id_to_index[req_id] for req_id in req_ids
        ]
        idx_mapping = self.input_buffers.idx_mapping
        idx_mapping.np[:num_reqs] = idx_mapping_list
        idx_mapping_np = idx_mapping.np[:num_reqs]
        idx_mapping = idx_mapping.copy_to_gpu(num_reqs)

        # Block tables: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(idx_mapping)

        prepare_inputs(
            idx_mapping_np,
            self.req_states.prefill_token_ids,
            self.req_states.num_computed_tokens,
            num_scheduled_tokens,
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            num_tokens,
        )

        query_start_loc = self.input_buffers.query_start_loc
        query_start_loc_gpu = query_start_loc.gpu[: num_reqs + 1]
        query_start_loc_np = query_start_loc.np[: num_reqs + 1]
        seq_lens_gpu = self.input_buffers.seq_lens.gpu[:num_reqs]
        seq_lens_np = self.input_buffers.seq_lens.np[:num_reqs]

        # Some input token ids are directly read from the last sampled tokens.
        combine_last_token_ids(
            self.input_buffers.input_ids.gpu,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc_gpu,
            seq_lens_gpu,
            self.req_states.prefill_len.copy_to_gpu(),
        )

        # Compute slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, self.input_buffers.positions.gpu[:num_tokens]
        )

        num_computed_tokens_cpu = torch.from_numpy(
            self.req_states.num_computed_tokens[idx_mapping_np]
        )

        # Logits indices to sample next token from.
        logits_indices = query_start_loc_gpu[1:] - 1

        # Layer name -> attention metadata.
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc=self.input_buffers.query_start_loc,
            seq_lens=self.input_buffers.seq_lens,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

        input_ids = self.input_buffers.input_ids.gpu[:num_tokens_after_padding]
        positions = self.input_buffers.positions.gpu[:num_tokens_after_padding]
        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            query_start_loc=query_start_loc_gpu,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens_gpu,
            seq_lens_np=seq_lens_np,
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        sampler_output = self.sampler.sample(logits, sampling_metadata)
        return sampler_output

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

        num_computed_tokens = self.req_states.num_computed_tokens[idx_mapping_np]
        prompt_lens = self.req_states.prompt_len[idx_mapping_np]
        # NOTE(woosuk): -1 because the last prompt token's hidden state is not
        # needed for prompt logprobs.
        includes_prompt = num_computed_tokens < prompt_lens - 1
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
        seq_lens = self.input_buffers.seq_lens.np[: input_batch.num_reqs]
        is_prompt_chunked = seq_lens < prompt_lens
        prefill_token_ids = self.req_states.prefill_token_ids
        query_start_loc = self.input_buffers.query_start_loc.np
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue
            if not is_prompt_chunked[i]:
                continue
            # The prompt is chunked. Get the next prompt token.
            req_idx = input_batch.idx_mapping_np[i]
            next_prompt_token = int(prefill_token_ids[req_idx, seq_lens[i]])
            idx = int(query_start_loc[i + 1] - 1)
            # Set the next prompt token.
            # NOTE(woosuk): This triggers a GPU operation.
            token_ids[idx] = next_prompt_token

        # NOTE(woosuk): We mask out logprobs for negative tokens.
        prompt_logprobs, prompt_ranks = compute_prompt_logprobs(
            torch.relu(token_ids),
            hidden_states[:n],
            self.model.compute_logits,
        )
        prompt_logprobs[:, 0].masked_fill_(token_ids < 0, 0)

        prompt_token_ids = token_ids.unsqueeze(-1)
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            start_idx = query_start_loc[i]
            end_idx = query_start_loc[i + 1]
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
        sampler_output: SamplerOutput,
        sampling_metadata: SamplingMetadata,
        prompt_logprobs_dict: dict[str, LogprobsTensors],
        input_batch: InputBatch,
    ) -> AsyncOutput | ModelRunnerOutput:
        # Store the last sampled token ids.
        self.req_states.last_sampled_tokens[input_batch.idx_mapping] = (
            sampler_output.sampled_token_ids
        )
        # Get the number of sampled tokens.
        # 0 if chunked-prefilling, 1 if not.
        idx_mapping_np = input_batch.idx_mapping_np
        is_chunked_prefilling = (
            input_batch.seq_lens_np < self.req_states.num_tokens[idx_mapping_np]
        )
        num_sampled_tokens = (~is_chunked_prefilling).astype(np.int32)
        # Increment the number of tokens.
        self.req_states.num_tokens[idx_mapping_np] += num_sampled_tokens
        # Increment the number of computed tokens.
        self.req_states.num_computed_tokens[idx_mapping_np] += (
            input_batch.num_scheduled_tokens
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled_tokens,
            copy_stream=self.output_copy_stream,
        )
        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
    ) -> AsyncOutput | ModelRunnerOutput:
        assert intermediate_tensors is None

        with async_barrier(
            self.input_prep_event if self.use_async_scheduling else None
        ):
            self.update_states(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens == 0:
                return EMPTY_MODEL_RUNNER_OUTPUT

            padded_num_tokens = self.cudagraph_manager.get_cudagraph_size(
                scheduler_output
            )
            use_cudagraph = padded_num_tokens is not None
            input_batch = self.prepare_inputs(
                scheduler_output,
                use_cudagraph,
                padded_num_tokens,
            )
            pos = input_batch.positions[input_batch.logits_indices]
            idx_mapping_np = input_batch.idx_mapping_np
            sampling_metadata = self.req_states.make_sampling_metadata(
                idx_mapping_np, pos
            )

        if self.lora_config:
            # Activate LoRA adapters.
            lora_inputs = self.req_states.make_lora_inputs(
                input_batch.req_ids,
                input_batch.idx_mapping_np,
                input_batch.num_scheduled_tokens,
            )
            self._set_active_loras(*lora_inputs)

        # Run model.
        if use_cudagraph:
            # Run CUDA graph.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            hidden_states = self.cudagraph_manager.run(padded_num_tokens)
        else:
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
            ):
                # Run PyTorch model in eager mode.
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=input_batch.positions,
                )

        sampler_output = self.sample(hidden_states, input_batch, sampling_metadata)
        prompt_logprobs_dict = self.compute_prompt_logprobs(hidden_states, input_batch)
        output = self.postprocess(
            sampler_output,
            sampling_metadata,
            prompt_logprobs_dict,
            input_batch,
        )
        return output
