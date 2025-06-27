# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import gc
import time
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.layer import Attention
from vllm.compilation.counter import compilation_counter
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture,
    prepare_communication_buffer_for_model)
from vllm.forward_context import (DPMetadata, get_forward_context,
                                  set_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import (has_step_pooler,
                                                   is_mixture_of_experts)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, LazyLoader, async_tensor_h2d, cdiv,
                        check_use_alibi, get_dtype_size,
                        is_pin_memory_available)
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .utils import (gather_mm_placeholders, initialize_kv_cache_for_kv_sharing,
                    sanity_check_mm_encoder_outputs, scatter_mm_placeholders)

if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501

    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile", globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile")

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
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_multimodal_model = model_config.is_multimodal_model
        self.is_pooling_model = model_config.pooler_config is not None
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Sampler
        self.sampler = Sampler()

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        self.attn_metadata_builders: list[AttentionMetadataBuilder] = []
        self.attn_backends: list[type[AttentionBackend]] = []
        # self.kv_cache_config: KVCacheConfig

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config,
                    device=self.device)  # type: ignore
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
        )

        self.use_cuda_graph = (
            self.vllm_config.compilation_config.level
            == CompilationLevel.PIECEWISE
            and self.vllm_config.compilation_config.use_cudagraph
            and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(self.compilation_config.cudagraph_capture_sizes))

        self.full_cuda_graph = self.compilation_config.full_cuda_graph

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)
            self.mrope_positions_np = self.mrope_positions_cpu.numpy()

        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> bool:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.

        Returns:
            True if the batch was reordered, False otherwise.
        """
        batch_reordered = self.attn_metadata_builders[0].reorder_batch(
            self.input_batch, scheduler_output)

        # For models with multiple KV cache groups, the groups should agree on
        # the same order of requests. We ensure this by only allowing the first
        # group to reorder the batch and asserting that all other groups do not
        # reorder the batch.
        for i in range(1, len(self.kv_cache_config.kv_cache_groups)):
            assert not self.attn_metadata_builders[i].reorder_batch(
                self.input_batch, scheduler_output)
        return batch_reordered

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties
        """
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params
            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_input in self.requests[req_id].mm_inputs:
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.extend(
                            mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.extend(
                            mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.extend(
                            mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.extend(
                            mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions_tensor(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                    )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_block_ids in zip(req_state.block_ids,
                                                    req_data.new_block_ids):
                    block_ids.extend(new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices.sort(reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        batch_reordered = self._may_reorder_batch(scheduler_output)

        if batch_changed or batch_reordered:
            self.input_batch.refresh_sampling_metadata()

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, Any], bool, torch.Tensor,
               Optional[SpecDecodeMetadata], np.ndarray]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            attention_cuda_graphs: whether attention can run in cudagraph
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size
            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc[num_reqs + 1:].fill_(
            self.query_start_loc_cpu[num_reqs].item())

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
        )

        attn_metadata: dict[str, Any] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = self.attn_metadata_builders[kv_cache_group_id]
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id],
                    kv_cache_group_spec.kv_cache_spec,
                    builder,
                )

            attn_metadata_i = (builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            ))

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        attention_cuda_graphs = all(
            b.can_run_in_cudagraph(common_attn_metadata)
            for b in self.attn_metadata_builders)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return (attn_metadata, attention_cuda_graphs, logits_indices,
                spec_decode_metadata, num_scheduled_tokens)

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len,
            self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (common_prefix_len // kv_cache_spec.block_size *
                             kv_cache_spec.block_size)
        use_sliding_window = (isinstance(kv_cache_spec, SlidingWindowSpec) or
                              (isinstance(kv_cache_spec, FullAttentionSpec)
                               and kv_cache_spec.sliding_window is not None))
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            num_sms=self.num_sms,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = len(req.prompt_token_ids)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions_cpu[:, dst_start:dst_end] = \
                    req.mrope_positions[:,src_start:src_end]

                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions_np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(
                grouped_mm_inputs, pin_memory=self.pin_memory)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_mm_inputs,
                device=self.device,
            )

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=len(grouped_mm_inputs),
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[req_id][input_id] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)
        return mm_embeds

    def get_model(self) -> nn.Module:
        return self.model

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ):
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.input_batch.req_id_to_index.items(),
                     key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.zeros_like(grammar_bitmask,
                                       shape=(logits.shape[0],
                                              grammar_bitmask.shape[1]))
        cumulative_index = 0
        seq = sorted(scheduler_output.structured_output_request_ids.items(),
                     key=lambda x: x[1])
        for req_id, _ in seq:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # Force use of the torch.compile implementation from xgrammar to work
        # around issues with the Triton kernel in concurrent structured output
        # scenarios. See PR #19565 and issues #19493, #18376 for details.
        xgr_torch_compile.apply_token_bitmask_inplace_torch_compile(
            logits,
            grammar_bitmask.to(self.device, non_blocking=True),
            indices=out_indices,
        )

    def sync_and_slice_intermediate_tensors(
            self, num_tokens: int, intermediate_tensors: IntermediateTensors,
            sync_self: bool) -> IntermediateTensors:

        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        enabled_sp = self.compilation_config.pass_config. \
            enable_sequence_parallelism
        if enabled_sp:
            # When sequence parallelism is enabled, we always pad num_tokens
            # to be a multiple of tensor_parallel_size (tp) earlier
            assert num_tokens % tp == 0
        is_residual_scattered = tp > 1 and enabled_sp \
            and num_tokens % tp == 0

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = "residual" and is_residual_scattered
                copy_len = num_tokens // tp if is_scattered else \
                    num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True)

        return IntermediateTensors({
            k:
            v[:num_tokens // tp]
            if k == "residual" and is_residual_scattered else v[:num_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    def eplb_step(self,
                  is_dummy: bool = False,
                  is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        assert is_mixture_of_experts(self.model)
        self.eplb_state.step(
            self.model,
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_log_balancedness,
        )

    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            # Early exit.
            return 0, None

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                                dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
        return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        finished_sending: Optional[set[str]],
        finished_recving: Optional[set[str]],
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        extracted_hidden_states = list(
            torch.split(hidden_states[:num_scheduled_tokens],
                        num_scheduled_tokens_np.tolist()))

        pooling_metadata = self.input_batch.pooling_metadata

        raw_pooler_output = self.model.pooler(
            hidden_states=extracted_hidden_states,
            pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        seq_lens = self.seq_lens[:self.input_batch.num_reqs]
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.cpu())
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:

        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)

        # Prepare the decoder inputs.
        (attn_metadata, attention_cuda_graphs, logits_indices,
         spec_decode_metadata,
         num_scheduled_tokens_np) = (self._prepare_inputs(scheduler_output))
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                from vllm.utils import round_up
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        # Some attention backends only support CUDA Graphs in pure decode.
        # If attention doesn't support CUDA Graphs for this batch, but we
        # compiled with full CUDA graphs, we have to skip them entirely.
        skip_cuda_graphs = self.full_cuda_graph and not attention_cuda_graphs

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                skip_cuda_graphs=skip_cuda_graphs,
        ):
            self.maybe_setup_kv_connector(scheduler_output)

            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            if self.input_batch.pooling_params:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np, finished_sending,
                                  finished_recving)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        if not self.speculative_config:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        elif self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self.generate_draft_token_ids(
                valid_sampled_token_ids, sampling_metadata)
        elif self.speculative_config.method == "medusa":
            assert isinstance(self.drafter, MedusaProposer)
            if max_gen_len == 1:
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        valid_sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1

                indices = torch.tensor(indices,
                                       device=sample_hidden_states.device)
                hidden_states = sample_hidden_states[indices]

            spec_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # TODO(woosuk): Refactor the loop.
            next_token_ids: list[int] = []
            for i, token_ids in enumerate(valid_sampled_token_ids):
                if token_ids:
                    # Common case.
                    next_token_id = token_ids[-1]
                else:
                    # Partial prefill (rare case).
                    # Get the next token id from the request state.
                    req_id = self.input_batch.req_ids[i]
                    req_state = self.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                               scheduler_output.num_scheduled_tokens[req_id])
                    next_token_id = req_state.get_token_id(seq_len)
                next_token_ids.append(next_token_id)
            next_token_ids = torch.tensor(next_token_ids,
                                          dtype=torch.int32,
                                          device=self.device)
            # At this moment, we assume all eagle layers belong to the same KV
            # cache group, thus using the same attention metadata.
            eagle_attn_metadata = attn_metadata[
                self.drafter.attn_layer_names[0]]

            # NOTE: deepseek_mtp uses MLA which does not have `block_table`
            if hasattr(eagle_attn_metadata, "block_table"):
                block_table = eagle_attn_metadata.block_table
            else:
                block_table = None

            if spec_decode_metadata is None:
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
                target_slot_mapping = eagle_attn_metadata.slot_mapping
                cu_num_tokens = eagle_attn_metadata.query_start_loc
            else:
                # TODO(woosuk): Refactor this.
                num_draft_tokens = spec_decode_metadata.num_draft_tokens
                num_rejected_tokens = [
                    n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                    for i, n in enumerate(num_draft_tokens)
                ]
                num_rejected_tokens_tensor = async_tensor_h2d(
                    num_rejected_tokens,
                    dtype=torch.int32,
                    target_device=self.device,
                    pin_memory=True)
                num_tokens = num_scheduled_tokens - sum(num_rejected_tokens)
                cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                    eagle_attn_metadata.query_start_loc,
                    num_rejected_tokens_tensor,
                    num_tokens,
                )
                target_token_ids = self.input_ids[token_indices]
                target_positions = positions[token_indices]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
                target_slot_mapping = eagle_attn_metadata.slot_mapping[
                    token_indices]
            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                target_slot_mapping=target_slot_mapping,
                next_token_ids=next_token_ids,
                cu_num_tokens=cu_num_tokens,
                block_table=block_table,
                sampling_metadata=sampling_metadata,
            )
            spec_token_ids = draft_token_ids.tolist()

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        self.eplb_step()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            num_nans_in_logits=num_nans_in_logits,
        )

    def kv_connector_no_forward(
            self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))

        if not finished_sending and not finished_recving:
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.finished_sending = finished_sending
        output.finished_recving = finished_recving
        return output

    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):
        # Update KVConnector with the KVConnector metadata forward().
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            assert isinstance(kv_connector, KVConnectorBase_V1)
            assert scheduler_output.kv_connector_metadata is not None
            kv_connector.bind_connector_metadata(
                scheduler_output.kv_connector_metadata)

            # Background KV cache transfers happen here.
            # These transfers are designed to be async and the requests
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def maybe_wait_for_kv_save() -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(
                scheduler_output.finished_req_ids)
        return None, None

    def generate_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = self.input_batch.req_ids[i]
            if not is_spec_decode_supported(req_id, self.input_batch):
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            if end_idx >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :end_idx])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            if not hasattr(self, "model"):
                logger.info("Loading model from scratch...")
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config,
                    model_config=self.model_config)
            else:
                logger.info(
                    "Model was already initialized. Loading weights inplace..."
                )
                model_loader.load_weights(self.model,
                                          model_config=self.model_config)
            if has_step_pooler(self.model):
                self.input_batch.logits_processing_needs_token_ids = True
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                self.model.set_aux_hidden_state_layers(
                    self.model.get_eagle3_aux_hidden_state_layers())
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)

        if is_mixture_of_experts(
                self.model) and self.parallel_config.enable_eplb:
            logger.info("EPLB is enabled for model %s.",
                        self.model_config.model)
            self.eplb_state = EplbState.build(
                self.model,
                self.device,
                self.parallel_config,
            )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc_np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states, None)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None
                    and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run 
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype)

            logger.debug("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)],
                            non_blocking=True)
            yield
            input_ids.fill_(0)

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        capture_attn_cudagraph: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None
        if capture_attn_cudagraph:
            attn_metadata = {}

            query_start_loc = self.query_start_loc[:num_reqs + 1]
            # Make sure max_model_len is used at the graph capture time.
            self.seq_lens_np[:num_reqs] = self.max_model_len
            self.seq_lens_np[num_reqs:] = 0
            self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                           non_blocking=True)
            seq_lens = self.seq_lens[:num_reqs]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=num_tokens,
            )

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):

                attn_metadata_i = self.attn_metadata_builders[
                    kv_cache_group_id].build_for_cudagraph_capture(
                        common_attn_metadata)
                for layer_name in kv_cache_group_spec.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None
            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp):
                outputs = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.
        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states, None)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            min_p=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            min_tokens={},
            logit_bias=[None for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
        )
        try:
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device)

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            target_logits = torch.randn(num_tokens,
                                        logits.shape[-1],
                                        device=self.device,
                                        dtype=logits.dtype)
            # NOTE(woosuk): Here, we should use int32 because the sampler uses
            # int32 for bonus_token_ids. If the dtype mismatches, re-compilation
            # will occur at runtime.
            bonus_token_ids = torch.zeros(num_reqs,
                                          device=self.device,
                                          dtype=torch.int32)
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                target_logits,
                bonus_token_ids,
                dummy_metadata,
            )
        return sampler_output

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        hidden_states_list = list(
            torch.split(hidden_states, num_scheduled_tokens_list))

        req_num_tokens = num_tokens // num_reqs

        dummy_metadata = PoolingMetadata(
            prompt_lens=torch.tensor([h.shape[0] for h in hidden_states_list],
                                     device=self.device),
            prompt_token_ids=torch.zeros((num_reqs, req_num_tokens),
                                         dtype=torch.int32,
                                         device=self.device),
            pooling_params=[PoolingParams()] * num_reqs)

        try:
            pooler_output = self.model.pooler(hidden_states=hidden_states_list,
                                              pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        return pooler_output

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            max_tokens_by_modality_dict = self.mm_registry \
                .get_max_tokens_per_item_by_nonzero_modality(self.model_config)
            dummy_data_modality, max_tokens_per_mm_item = max(
                max_tokens_by_modality_dict.items(), key=lambda item: item[1])

            # Check how many items of this modality can be supported by
            # the encoder budget.
            encoder_budget = min(self.max_num_encoder_input_tokens,
                                 self.encoder_cache_size)

            max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                                   max_tokens_per_mm_item)

            # Check how many items of this modality can be supported by
            # the decoder budget.
            max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
                self.model_config)[dummy_data_modality]

            # NOTE: We do not consider max_num_batched_tokens on purpose
            # because the multimodal embeddings can be generated in advance
            # and chunked prefilled.
            max_num_mm_items_decoder_budget = self.max_num_reqs * \
                max_mm_items_per_req

            max_num_mm_items = min(max_num_mm_items_encoder_budget,
                                   max_num_mm_items_decoder_budget)

            logger.info(
                "Encoder cache will be initialized with a budget of %s tokens,"
                " and profiled with %s %s items of the maximum feature size.",
                encoder_budget, max_num_mm_items, dummy_data_modality)

            # Create dummy batch of multimodal inputs.
            dummy_mm_kwargs = self.mm_registry.get_decoder_dummy_data(
                model_config=self.model_config,
                seq_len=self.max_num_tokens,
                mm_counts={
                    dummy_data_modality: 1
                },
            ).multi_modal_data

            batched_dummy_mm_inputs = MultiModalKwargs.batch(
                [dummy_mm_kwargs] * max_num_mm_items,
                pin_memory=self.pin_memory)
            batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_dummy_mm_inputs,
                device=self.device,
            )

            # Run multimodal encoder.
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)

            sanity_check_mm_encoder_outputs(
                dummy_encoder_outputs,
                expected_num_items=max_num_mm_items,
            )

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states \
            = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "set -O %s and ensure `use_cudagraph` was not manually set to "
                "False", CompilationLevel.PIECEWISE)
            return

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            full_cg = self.full_cuda_graph
            for num_tokens in tqdm(reversed(self.cudagraph_batch_sizes),
                                   desc="Capturing CUDA graphs",
                                   total=len(self.cudagraph_batch_sizes)):
                # We skip EPLB here since we don't want to record dummy metrics
                for _ in range(
                        self.compilation_config.cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens,
                                    capture_attn_cudagraph=full_cg,
                                    skip_eplb=True)
                self._dummy_run(num_tokens,
                                capture_attn_cudagraph=full_cg,
                                skip_eplb=True)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_backends) == 0 and len(
            self.attn_metadata_builders
        ) == 0, "Attention backends are already initialized"
        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if isinstance(kv_cache_spec, AttentionSpec):
                attn_backend_i = get_attn_backend(
                    kv_cache_spec.head_size,
                    self.dtype,
                    kv_cache_spec.dtype,
                    kv_cache_spec.block_size,
                    self.model_config.is_attention_free,
                    use_mla=kv_cache_spec.use_mla,
                )
                if attn_backend_i is None:
                    error_msg = (f"Error with get_attn_backend: "
                                 f"{kv_cache_spec.head_size=}, "
                                 f"{self.dtype=}, {kv_cache_spec.dtype=}, "
                                 f"{kv_cache_spec.block_size=}, "
                                 f"{self.model_config.is_attention_free=}, "
                                 f"{kv_cache_spec.use_mla=}")
                    logger.error(error_msg)
                    raise NotImplementedError(
                        "Non-Attention backend is not supported by V1 "
                        "GPUModelRunner.")
            elif isinstance(kv_cache_spec, MambaSpec):
                attn_backend_i = Mamba2AttentionBackend
            else:
                raise ValueError(
                    f"Unknown KV cache spec type: {type(kv_cache_spec)}")

            block_table_i = self.input_batch.block_table[i]
            attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
                weakref.proxy(self),
                kv_cache_spec,
                block_table_i,
            )

            if (self.full_cuda_graph
                    and not attn_metadata_builder_i.full_cudagraph_supported):
                raise ValueError(
                    f"Full CUDAGraph not supported for "
                    f"{attn_backend_i.__name__}. Turn off CompilationConfig."
                    f"full_cuda_graph or use a different attention backend.")

            self.attn_backends.append(attn_backend_i)
            self.attn_metadata_builders.append(attn_metadata_builder_i)

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
            )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device=self.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            layer_names.update(group.layer_names)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        for i, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            for layer_name in kv_cache_group_spec.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = self.attn_backends[
                            i].get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    dtype = kv_cache_spec.dtype
                    state_tensors = []
                    start_pos = 0
                    for shape in kv_cache_spec.shapes:
                        target_shape = (num_blocks, *shape)
                        size_in_bytes = np.prod(shape) * get_dtype_size(
                            dtype) * num_blocks
                        tensor = raw_tensor[start_pos:start_pos +
                                            size_in_bytes]
                        tensor = tensor.view(dtype).view(target_shape)
                        state_tensors.append(tensor)
                        start_pos += size_in_bytes
                    assert start_pos == raw_tensor.numel()
                    kv_caches[layer_name] = tuple(state_tensors)
                else:
                    raise NotImplementedError
        return kv_caches

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Setup `kv_cache_config` and `kv_caches` for models
        # with cross-layer KV sharing
        if self.shared_kv_cache_layers:
            initialize_kv_cache_for_kv_sharing(
                self.shared_kv_cache_layers,
                kv_cache_config.kv_cache_groups,
                kv_caches,
            )

        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches)
        return kv_caches

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        self.may_reinitialize_input_batch(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer :=
                    attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            # TODO: Support other attention modules, e.g., cross-attention
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        mamba_layers = get_layers_from_vllm_config(self.vllm_config,
                                                   MambaMixer2)
        if len(mamba_layers) > 0:
            if self.vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if not self.vllm_config.model_config.enforce_eager:
                raise NotImplementedError(
                    "Mamba with cuda graph is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len
            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtype=self.kv_cache_dtype,
                    block_size=max_model_len)
        return kv_cache_spec
