# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, is_pin_memory_available)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.sampler import SamplerOutput
from vllm.v1.worker.gpu.attn_utils import (get_kv_cache_spec,
                                           init_attn_backend, init_kv_cache)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import (InputBatch, InputBuffers,
                                            prepare_inputs)
from vllm.v1.worker.gpu.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import bind_kv_cache

logger = init_logger(__name__)


class GPUModelRunner:

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
                self.cache_config.cache_dtype]
        self.is_pooling_model = False

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

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
            device=self.device,
            pin_memory=self.pin_memory,
        )
        self.sampler = Sampler()

    def load_model(self, *args, **kwargs) -> None:
        time_before_load = time.perf_counter()
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.model_config,
            )
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    m.consumed_memory / GiB_bytes,
                    time_after_load - time_before_load)

    def profile_run(self):
        pass

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

        kv_caches = init_kv_cache(
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        self.kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )

    def _dummy_run(self, num_tokens: int, *args, **kwargs) -> None:
        return None, None

    def _dummy_sampler_run(self, hidden_states: torch.Tensor, *args,
                           **kwargs) -> None:
        return None

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        # for req_id in scheduler_output.preempted_req_ids:
        #     self.req_states.remove_request(req_id)
        for req_id in scheduler_output.finished_req_ids:
            self.req_states.remove_request(req_id)

        # TODO(woosuk): Change SchedulerOutput.
        req_indices: list[int] = []
        cu_num_new_blocks = tuple(
            [0] for _ in range(self.block_tables.num_kv_cache_groups))
        new_block_ids = tuple(
            [] for _ in range(self.block_tables.num_kv_cache_groups))
        overwrite: list[bool] = []

        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self.req_states.add_request(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                sampling_params=new_req_data.sampling_params,
            )

            req_index = self.req_states.req_id_to_index[req_id]
            req_indices.append(req_index)
            for i, block_ids in enumerate(new_req_data.block_ids):
                x = cu_num_new_blocks[i][-1]
                cu_num_new_blocks[i].append(x + len(block_ids))
                new_block_ids[i].extend(block_ids)
            overwrite.append(True)

        # Update the states of the running/resumed requests.
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

            self.req_states.num_computed_tokens[req_index] = (
                cached_reqs.num_computed_tokens[i])

        if req_indices:
            self.block_tables.append_block_ids(
                req_indices=req_indices,
                cu_num_new_blocks=cu_num_new_blocks,
                new_block_ids=new_block_ids,
                overwrite=overwrite,
            )

    def prepare_inputs(self, scheduler_output: SchedulerOutput) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(scheduler_output.num_scheduled_tokens,
                         key=scheduler_output.num_scheduled_tokens.get)
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32)

        # TODO(woosuk): Support CUDA graphs.
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

        input_ids = self.input_buffers.input_ids
        positions = self.input_buffers.positions
        query_start_loc = self.input_buffers.query_start_loc
        seq_lens = self.input_buffers.seq_lens
        prepare_inputs(
            idx_mapping_np,
            self.req_states.token_ids,
            self.req_states.num_computed_tokens,
            num_scheduled_tokens,
            input_ids.np,
            positions.np,
            query_start_loc.np,
            seq_lens.np,
        )
        input_ids.copy_to_gpu(num_tokens)
        positions.copy_to_gpu(num_tokens)

        # NOTE(woosuk): We should copy the whole query_start_loc and seq_lens
        # tensors from CPU to GPU, because they may include paddings needed
        # for full CUDA graph mode.
        query_start_loc.copy_to_gpu()
        query_start_loc_cpu = query_start_loc.cpu[:num_reqs + 1]
        query_start_loc_gpu = query_start_loc.gpu[:num_reqs + 1]
        max_query_len = int(num_scheduled_tokens.max())

        seq_lens.copy_to_gpu()
        seq_lens_cpu = seq_lens.cpu[:num_reqs]
        seq_lens_np = seq_lens.np[:num_reqs]
        max_seq_len = int(seq_lens_np.max())
        seq_lens_gpu = seq_lens.gpu[:num_reqs]

        num_computed_tokens_np = self.req_states.num_computed_tokens[
            idx_mapping_np]
        num_computed_tokens_cpu = torch.from_numpy(num_computed_tokens_np)
        is_chunked_prefilling = (seq_lens_np
                                 < self.req_states.num_tokens[idx_mapping_np])

        # Slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc_gpu, positions.gpu[:num_tokens])
        logits_indices = query_start_loc_gpu[1:] - 1
        num_logits_indices = logits_indices.size(0)

        # Layer name -> attention metadata.
        attn_metadata: dict[str, Any] = {}
        for i, kv_cache_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_table = block_tables[i]
            slot_mapping = slot_mappings[i]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc_gpu,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_gpu,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                logits_indices_padded=None,
                num_logits_indices=num_logits_indices,
                causal=True,
                encoder_seq_lens=None,
            )

            attn_metadata_builder = self.attn_metadata_builders[i]
            metadata = attn_metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in kv_cache_spec.layer_names:
                attn_metadata[layer_name] = metadata

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            is_chunked_prefilling=is_chunked_prefilling,
            input_ids=input_ids.gpu,
            positions=positions.gpu,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
        )

    def sample(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        pos = input_batch.positions[input_batch.logits_indices]
        sampling_metadata = self.req_states.make_sampling_metadata(
            input_batch.idx_mapping_np, pos)
        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return sampler_output

    def postprocess(
        self,
        sampler_output: SamplerOutput,
        input_batch: InputBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Get the number of sampled tokens.
        # 0 if chunked-prefilling, 1 if not.
        is_chunked_prefilling = input_batch.is_chunked_prefilling
        num_sampled_tokens = (~is_chunked_prefilling).astype(np.int32)

        # Update the token IDs and the number of tokens.
        sampled_token_ids_cpu = sampler_output.sampled_token_ids.cpu()
        sampled_token_ids_np = sampled_token_ids_cpu.numpy()
        self.req_states.append_token_ids(
            input_batch.idx_mapping_np,
            sampled_token_ids_np,
            num_sampled_tokens=num_sampled_tokens,
        )
        return sampled_token_ids_np, num_sampled_tokens

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ):
        self.update_states(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT

        input_batch = self.prepare_inputs(scheduler_output)
        num_tokens = input_batch.num_tokens_after_padding

        with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
        ):
            hidden_states = self.model(
                input_ids=input_batch.input_ids[:num_tokens],
                positions=input_batch.positions[:num_tokens],
            )

        # Compute logits to sample next tokens.
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        sampler_output = self.sample(logits, input_batch)
        sampled_token_ids_np, num_sampled_tokens = self.postprocess(
            sampler_output, input_batch)
        req_id_to_index = {
            req_id: i
            for i, req_id in enumerate(input_batch.req_ids)
        }
        return ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids_np.tolist(),
            logprobs=sampler_output.logprobs_tensors,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=None,
            num_nans_in_logits=None,
        )
