# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import itertools
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.distributed.parallel_state import (get_pp_group, get_tp_group,
                                             graph_capture,
                                             is_global_first_rank)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LazyLoader, check_use_alibi,
                        is_pin_memory_available, round_up)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, CommonAttentionMetadata,
    create_fast_prefill_custom_backend)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        KVCacheConfig, KVCacheSpec)
# yapf: enable
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             SamplerOutput)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import InputBatch, prepare_inputs
from vllm.v1.worker.gpu_worker_states import RequestState
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .utils import AttentionGroup, MultiModalBudget, bind_kv_cache

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self._async_copy_ready_event = torch.cuda.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._sampled_token_ids_cpu = self._sampled_token_ids.to(
                'cpu', non_blocking=True)
            self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        
        This function blocks until the copy is finished.
        """
        self._async_copy_ready_event.synchronize()

        # Release the device tensor once the copy has completed
        del self._sampled_token_ids

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


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

        self.is_pooling_model = (model_config.runner_type == 'pooling')
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model)

        self.max_model_len = model_config.max_model_len
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend
            == "external_launcher" and len(get_pp_group().ranks) > 0)

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = self.mm_registry.\
                get_encdec_max_encoder_len(model_config)
        else:
            self.max_encoder_len = 0

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

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
        self.max_num_cached_reqs = 2 * self.max_num_reqs
        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            max_num_cached_reqs=self.max_num_cached_reqs,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.async_output_copy_stream = torch.cuda.Stream() if \
            self.use_async_scheduling else None

        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        if self.compilation_config.cudagraph_capture_sizes and \
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self.cudagraph_batch_sizes = list(
                reversed(self.compilation_config.cudagraph_capture_sizes))

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int64)
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1,
                                                 dtype=torch.int32)
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(self.max_num_tokens,
                                               self.hidden_size,
                                               dtype=self.dtype,
                                               numpy=False)
        self.num_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                  dtype=torch.int32)
        self.num_accepted_tokens = self._make_buffer(self.max_num_reqs,
                                                     dtype=torch.int64)

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
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64)

        # CUDA event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: Optional[torch.cuda.Event] = None
        if self.use_async_scheduling:
            self.prepare_inputs_event = torch.cuda.Event()
            # Start in a completed state.
            self.prepare_inputs_event.record(torch.cuda.default_stream())

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device)

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = MultiModalBudget(
            self.model_config,
            self.scheduler_config,
            self.mm_registry,
        ) if self.supports_mm_inputs else None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]],
                                              torch.Tensor]] = None
        self.transfer_event = torch.cuda.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory)

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> InputBatch:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = len(scheduler_output.num_scheduled_tokens)

        # batch_idx -> req_id
        req_ids = sorted(scheduler_output.num_scheduled_tokens,
                         key=scheduler_output.num_scheduled_tokens.get)
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32)

        # batch_idx -> req_idx
        idx_mapping_list = [
            self.req_states.req_id_to_index[req_id] for req_id in req_ids
        ]
        self.idx_mapping.np[:num_reqs] = idx_mapping_list
        idx_mapping_np = self.idx_mapping.np[:num_reqs]
        idx_mapping = self.idx_mapping.copy_to_gpu(num_reqs)
        # req_id -> batch_idx
        req_id_to_batch_idx = {req_id: i for i, req_id in enumerate(req_ids)}

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        block_tables = self.block_tables.compute_block_tables(idx_mapping)

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[i] for i in req_ids],
            dtype=np.int32)

        prepare_inputs(
            idx_mapping_np,
            self.req_states.token_ids.np,
            self.req_states.num_computed_tokens.np,
            num_scheduled_tokens,
            self.input_ids.np,
            self.query_start_loc.np,
            self.seq_lens.np,
            self.positions.np,
        )
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)

        # NOTE(woosuk): We should copy the whole query_start_loc and seq_lens
        # tensors from CPU to GPU, because they may include paddings needed
        # for full CUDA graph mode.
        self.query_start_loc.copy_to_gpu()
        self.seq_lens.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]
        max_query_len = int(num_scheduled_tokens.max())
        seq_lens = self.seq_lens.gpu[:num_reqs]
        max_seq_len = int(self.seq_lens.np[:num_reqs].max())

        # Compute the slot mappings on GPUs.
        slot_mappings = self.block_tables.compute_slot_mappings(
            query_start_loc, self.positions.gpu[:total_num_scheduled_tokens])

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(req_ids, num_scheduled_tokens)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            spec_decode_metadata = self._prepare_spec_decode_metadata(
                req_ids,
                scheduler_output.scheduled_spec_decode_tokens,
                query_start_loc,
            )
            logits_indices = spec_decode_metadata.logits_indices
            self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_draft_tokens.np[num_reqs:].fill(0)
            self.num_draft_tokens.copy_to_gpu()

        logits_indices_padded = None
        if self.cache_config.kv_sharing_fast_prefill:
            logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices)

        # Used in the below loop.
        query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_np = self.req_states.num_computed_tokens.np[
            idx_mapping_np]
        num_computed_tokens_cpu = torch.from_numpy(num_computed_tokens_np)
        spec_decode_common_attn_metadata = None
        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        attn_metadata: dict[str, Any] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
                num_common_prefix_blocks = 0
            else:
                blk_table_tensor = block_tables[kv_cache_group_id]
                slot_mapping = slot_mappings[kv_cache_group_id]
                num_common_prefix_blocks = (
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id])

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                logits_indices_padded=logits_indices_padded,
                num_logits_indices=logits_indices.size(0),
                causal=True,
                encoder_seq_lens=encoder_seq_lens,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = attn_group.metadata_builder
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_common_prefix_blocks,
                        kv_cache_group_spec.kv_cache_spec,
                        builder,
                    )

                extra_attn_metadata_args = {}
                if use_spec_decode and isinstance(builder,
                                                  GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.
                        gpu[:num_reqs],
                        num_draft_tokens=self.num_draft_tokens.gpu[:num_reqs],
                    )

                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return InputBatch(
            req_ids=req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            req_id_to_batch_idx=req_id_to_batch_idx,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_reqs=num_reqs,
            total_num_tokens=total_num_scheduled_tokens,
            max_query_len=max_query_len,
            attn_metadata=attn_metadata,
            spec_decode_metadata=spec_decode_metadata,
            spec_decode_common_attn_metadata=spec_decode_common_attn_metadata,
            logits_indices=logits_indices,
        )

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and not envs.VLLM_DISABLE_PAD_FOR_CUDAGRAPH
                and hasattr(self, "cudagraph_batch_sizes")
                and self.cudagraph_batch_sizes
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use CUDA graphs.
            # Add padding to the batch size.
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (self.compilation_config.pass_config.enable_sequence_parallelism
                and tp_size > 1):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: InputBatch,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[int, int, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], torch.Tensor,
               Optional[IntermediateTensors], dict[str, Any]]:

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_input_tokens)
        num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if (self.supports_mm_inputs and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        if (self.model_config.is_encoder_decoder
                and scheduler_output.scheduled_encoder_inputs):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )

    def _sample(
        self,
        logits: Optional[torch.Tensor],
        input_batch: InputBatch,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = input_batch.sampling_metadata
        spec_decode_metadata = input_batch.spec_decode_metadata
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
            self._update_states_after_model_execute(output_token_ids)

        return sampler_output

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
            dict[str, int],
            Optional[LogprobsLists],
            list[list[int]],
            dict[str, Optional[LogprobsTensors]],
            list[str],
            dict[str, int],
            list[int],
    ]:
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

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = list(discard_sampled_tokens_req_indices)
            invalid_req_indices_set = set(invalid_req_indices)
            assert sampled_token_ids.shape[-1] == 1

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            self.input_batch.prev_sampled_token_ids = \
                sampled_token_ids
            self.input_batch.prev_sampled_token_ids_invalid_indices = \
                invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if \
                    req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.input_batch.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect logprobs for "
                    "prompt tokens, tokens, please disable it when the requests"
                    " need prompt logprobs")

            if self.prepare_inputs_event is not None:
                # Ensure prior step has finished with reused CPU tensors.
                self.prepare_inputs_event.synchronize()
            try:
                # Prepare the decoder inputs.
                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len) = self._prepare_inputs(scheduler_output)

            finally:
                if self.prepare_inputs_event is not None:
                    self.prepare_inputs_event.record()

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(scheduler_output, input_batch,
                                 intermediate_tensors)

            uniform_decode = (max_query_len
                              == self.uniform_decode_query_len) and (
                                  num_scheduled_tokens
                                  == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
        ), record_function_or_nullcontext("Forward"),
              self.maybe_get_kv_connector_output(scheduler_output) as
              kv_connector_output):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("Postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(hidden_states, num_scheduled_tokens,
                                        num_scheduled_tokens_np)
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual":
                        not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors)
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states,
                                                       None)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, input_batch)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                       logits, hidden_states,
                                       num_scheduled_tokens)

        if self.speculative_config:
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    valid_sampled_token_ids,
                    input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
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
            num_tokens = num_scheduled_tokens[req_id]

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
            offset = self.query_start_loc.np[req_idx].item()
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

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        force_attention: bool = False,
        uniform_decode: bool = False,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # Padding for DP
        num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
        num_tokens += num_pad

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [
                num_prefill_tokens
            ]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            num_reqs = num_tokens // max_query_len
            assert num_reqs <= max_num_reqs, \
                "Do not capture num_reqs > max_num_reqs for uniform batch"
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] += num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)

        attn_metadata: Optional[dict[str, Any]] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                # Make sure max_model_len is used at the graph capture time.
                seq_lens = self.max_model_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.block_table[
                        kv_cache_group_id].get_device_tensor()[:num_reqs],
                    slot_mapping=self.input_batch.
                    block_table[kv_cache_group_id].slot_mapping[:num_tokens],
                    causal=True)

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    attn_metadata_i = attn_group.metadata_builder\
                        .build_for_cudagraph_capture(common_attn_metadata)
                    for layer_name in kv_cache_group_spec.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if (self.supports_mm_inputs
                    and not self.model_config.is_encoder_decoder):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

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
            if cudagraph_runtime_mode == CUDAGraphMode.NONE:
                batch_descriptor = None
            else:
                # filter out the valid batch descriptor
                _cg_mode, batch_descriptor = \
                    self.cudagraph_dispatcher.dispatch(
                        BatchDescriptor(num_tokens=num_tokens,
                                        uniform_decode=uniform_decode))
                # sanity check
                assert cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
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
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
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

    def profile_run(self) -> None:
        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states \
            = self._dummy_run(self.max_num_tokens, is_profile=True)
        output = self._dummy_sampler_run(last_hidden_states)
        del hidden_states, output
        gc.collect()

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`")
            return 0
        else:
            self.initialize_cudagraph_capture()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            cudagraph_mode = self.compilation_config.cudagraph_mode
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False)

            # Capture full cudagraph for uniform decode batches if we have
            # dont already have full mixed prefill-decode cudagraphs
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                cudagraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    x <= max_num_tokens and x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True)

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
        return cuda_graph_size

    def _capture_cudagraphs(self, compilation_cases: list[int],
                            cudagraph_runtime_mode: CUDAGraphMode,
                            uniform_decode: bool):
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL,
                                        CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name))
        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = (
                    cudagraph_runtime_mode == CUDAGraphMode.FULL)
                self._dummy_run(num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode,
                                skip_eplb=True,
                                remove_lora=False)
            self._dummy_run(num_tokens,
                            cudagraph_runtime_mode=cudagraph_runtime_mode,
                            uniform_decode=uniform_decode,
                            skip_eplb=True,
                            remove_lora=False)
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        def get_attn_backends_for_layers(
                layer_names: list[str]
        ) -> dict[type[AttentionBackend], list[str]]:
            layers = get_layers_from_vllm_config(self.vllm_config,
                                                 AttentionLayerBase,
                                                 layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,
                    )

                key = attn_backend.full_cls_name()
                attn_backends[key] = attn_backend
                attn_backend_layers[key].append(layer_name)
            return {
                attn_backends[k]: v
                for k, v in attn_backend_layers.items()
            }

        def create_attn_groups(
            attn_backends_map: dict[AttentionBackend, list[str]],
            kv_cache_spec: KVCacheSpec,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for attn_backend, layer_names in attn_backends_map.items():
                attn_metadata_builder_i = attn_backend.get_builder_cls()(
                    kv_cache_spec,
                    layer_names,
                    self.vllm_config,
                    self.device,
                )
                attn_group = AttentionGroup(attn_backend,
                                            attn_metadata_builder_i,
                                            layer_names)
                attn_groups.append(attn_group)
            return attn_groups

        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            attn_backends = get_attn_backends_for_layers(
                kv_cache_group_spec.layer_names)
            self.attn_groups.append(
                create_attn_groups(attn_backends, kv_cache_spec))

    def initialize_cudagraph_capture(self) -> None:
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.metadata_builder
            if builder.cudagraph_support.value < min_cg_support.value:
                min_cg_support = builder.cudagraph_support
                min_cg_builder_name = builder.__class__.__name__

        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        # check cudagraph for mixed batch is supported
        if cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL \
            and min_cg_support != AttentionCGSupport.ALWAYS:
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                   f"with {min_cg_builder_name} backend (support: "
                   f"{min_cg_support})")
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += "; please try cudagraph_mode=PIECEWISE, and "\
                    "make sure compilation level is piecewise"
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_DECODE_ONLY
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and self.uniform_decode_query_len > 1 and min_cg_support.value
                < AttentionCGSupport.UNIFORM_BATCH.value):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                   f" with spec-decode for attention backend "
                   f"{min_cg_builder_name} (support: {min_cg_support})")
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.PIECEWISE
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.NONE
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if cudagraph_mode.has_full_cudagraphs() \
            and min_cg_support == AttentionCGSupport.NEVER:
            raise ValueError(f"CUDAGraphMode.{cudagraph_mode.name} is not "
                             f"supported with {min_cg_builder_name} backend ("
                             f"support:{min_cg_support}) "
                             "; please try cudagraph_mode=PIECEWISE, "
                             "and make sure compilation level is piecewise")

        # Trigger cudagraph dispatching keys initialization here (after
        # initializing attn backends).
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

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
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(
            self) -> Iterator[tuple[KVCacheSpec, AttentionGroup]]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for kv_cache_spec_id, attn_groups in enumerate(self.attn_groups):
            for attn_group in attn_groups:
                yield self.kv_cache_config.kv_cache_groups[
                    kv_cache_spec_id].kv_cache_spec, attn_group

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
        has_attn, has_mamba = False, False
        for kv_cache_spec, group in self._kv_cache_spec_attn_group_iterator():
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = \
                            attn_backend.get_kv_cache_stride_order()
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
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[:sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
