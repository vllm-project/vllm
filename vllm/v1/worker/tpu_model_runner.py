# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import gc
import time
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

# TPU XLA related
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

import vllm.envs as envs
from vllm.attention import Attention
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import MLAAttention
from vllm.attention.layers.chunked_local_attention import ChunkedLocalAttention
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import (
    ParallelConfig,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.tpu import TPUModelLoader
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.math_utils import cdiv, prev_power_of_2
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.pallas import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasAttentionBackend,
    PallasMetadata,
    get_page_size_bytes,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata
from vllm.v1.sample.tpu.sampler import Sampler as TPUSampler
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.tpu_input_batch import CachedRequestState, InputBatch

from .utils import (
    MultiModalBudget,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    sanity_check_mm_encoder_outputs,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)

INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selecting hidden states for each request
# 3. sampler
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class TPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        original_parallel_config: ParallelConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.original_parallel_config = original_parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        # SPMD Related
        self.use_spmd = envs.VLLM_XLA_USE_SPMD
        if self.use_spmd:
            num_devices = xr.global_runtime_device_count()
            mesh_shape = (num_devices, 1)
            device_ids = np.array(range(num_devices))
            self.mesh = xs.Mesh(device_ids, mesh_shape, ("x", "y"))

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            model_dtype = self.dtype
            if isinstance(model_dtype, str):
                self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            else:
                self.kv_cache_dtype = model_dtype
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.num_blocks_per_most_len_req = (
            cdiv(self.most_model_len, self.block_size)
            if self.most_model_len is not None
            else None
        )
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP,
        )
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, "attention"
        )
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()

        if self.lora_config is not None:
            self.vocab_size += self.lora_config.lora_extra_vocab_size

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        self._num_slices_per_kv_cache_update_block = (
            _get_num_slices_per_kv_cache_update_block(
                get_page_size_bytes(
                    block_size=self.block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    kv_cache_dtype=self.kv_cache_dtype,
                )
            )
        )

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # mm_hash -> encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Initialize input batch early to avoid AttributeError in _update_states
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
        )

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        self.input_ids_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )

        self.positions_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )
        self.positions_np = self.positions_cpu.numpy()
        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
        )
        # adjust num_reqs to avoid SMEM OOM.
        self.num_reqs_most_model_len = (
            min(
                PallasAttentionBackend.get_max_num_seqs(
                    self.most_model_len, self.block_size
                ),
                self.max_num_reqs,
            )
            if self.most_model_len is not None
            else None
        )
        self.num_reqs_max_model_len = min(
            PallasAttentionBackend.get_max_num_seqs(
                self.max_model_len, self.block_size
            ),
            self.max_num_reqs,
        )
        self.query_start_loc_cpu = torch.zeros(
            self.max_num_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            self.is_mm_embed_cpu = torch.zeros(
                self.max_num_tokens,
                dtype=torch.bool,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory
        )

        self.mm_budget = (
            MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            )
            if self.supports_mm_inputs
            else None
        )

        if not self.use_spmd:
            self.sample_from_logits_func = torch.compile(
                self.sample_from_logits,
                backend="openxla",
                fullgraph=True,
                dynamic=False,
            )
        else:
            self.sample_from_logits_func = self.sample_from_logits

        # For passing scheduler_output between successive
        # execute_model() and sample_tokens() calls.
        self.scheduler_output: SchedulerOutput | None = None
        self.mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None

    def reset_mm_cache(self) -> None:
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        total_cached_graphs = xr.get_num_cached_compilation_graph()
        new_compiled_graphs = total_cached_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0:
            return

        logger.info(
            "Add new %d compiled XLA graphs due to %s", new_compiled_graphs, case_str
        )
        self.num_xla_graphs += new_compiled_graphs

    def _verify_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        curr_cached_graph = xr.get_num_cached_compilation_graph()
        assert self.num_xla_graphs == curr_cached_graph, (
            "Recompilation after warm up is detected during {}."
            " num_xla_graphs = {} curr_cached_graph = {}".format(
                case_str, self.num_xla_graphs, curr_cached_graph
            )
        )

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

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
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

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
            assert new_req_data.sampling_params is not None, (
                "Pooling is not supported in TPU yet"
            )
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            # Fill the empty index or append to the end
            req_index = removed_req_indices.pop() if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            # Classic Attention path
            if isinstance(attn_module, Attention):
                if (
                    kv_tgt_layer := attn_module.kv_sharing_target_layer_name
                ) is not None:
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if attn_module.attn_type == AttentionType.DECODER:
                    if isinstance(attn_module, ChunkedLocalAttention):
                        logger.warning_once(
                            "Using irope in Pallas is not supported yet, it "
                            "will fall back to global attention for long context."
                        )
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            sliding_window=attn_module.sliding_window,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                        )
                elif attn_module.attn_type in (
                    AttentionType.ENCODER,
                    AttentionType.ENCODER_ONLY,
                ):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")
            # MLAAttention path
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                )
            else:
                continue

        return kv_cache_spec

    def _get_slot_mapping_metadata(
        self, num_reqs, num_scheduled_tokens_per_req
    ) -> np.ndarray:
        """
        Computes metadata for mapping slots to blocks in the key-value (KV)
        cache for a batch of requests.

        This function determines, for each request in the batch, how the
        scheduled tokens are distributed across memory blocks, and generates
        metadata needed to map slices of tokens to their corresponding positions
        in the KV cache.

        Args:
            num_reqs (int): Number of requests in the current batch.
            num_scheduled_tokens_per_req (int or np.ndarray): Number of tokens
                to be scheduled for each request.

        Returns:
            np.ndarray: A 2D array of shape (total_block_len, 3), where each row
                contains:
                - kv_cache_start_index (int): The starting index in the KV cache
                  for the corresponding slice.
                - new_kv_start_index (int): The starting index in the new KV
                  cache for the corresponding slice.
                - slice_len (int): The length of the slice.
        """
        slices_start = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        slices_end = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens_per_req
        )
        local_block_start_idx = slices_start // self.block_size
        local_block_end_idx = (slices_end - 1) // self.block_size
        no_repeat_req_indices = self.arange_np[:num_reqs]
        global_block_start_idx = (
            no_repeat_req_indices * self.max_num_blocks_per_req + local_block_start_idx
        )
        block_lens = local_block_end_idx - local_block_start_idx + 1
        global_block_start_idx = np.repeat(global_block_start_idx, block_lens)
        slice_arange = np.concatenate([self.arange_np[:n] for n in block_lens])
        global_block_indices = global_block_start_idx + slice_arange
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[global_block_indices].numpy()
        total_block_len = np.sum(block_lens)
        slot_mapping_slices = np.repeat(
            np.array([[0, self.block_size]], dtype=np.int32), total_block_len, axis=0
        )
        cu_block_lens = np.zeros(len(block_lens) + 1, dtype=np.int32)
        np.cumsum(block_lens, out=cu_block_lens[1:])
        for req_idx in range(num_reqs):
            slot_mapping_slices[cu_block_lens[req_idx]][0] = (
                slices_start[req_idx] % self.block_size
            )
            slot_mapping_slices[cu_block_lens[req_idx + 1] - 1][1] = (
                slices_end[req_idx] - 1
            ) % self.block_size + 1
        slice_lens = slot_mapping_slices[:, 1] - slot_mapping_slices[:, 0]
        cu_slices_lens = np.zeros(len(slice_lens) + 1, dtype=np.int32)
        np.cumsum(slice_lens, out=cu_slices_lens[1:])
        kv_cache_start_indices = slot_mapping_slices[:, 0] + (
            block_numbers * self.block_size
        )
        new_kv_start_indices = cu_slices_lens[:-1]
        slot_mapping_metadata = np.stack(
            [kv_cache_start_indices, new_kv_start_indices, slice_lens], axis=1
        )
        return slot_mapping_metadata

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput", start_index: int):
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs

        # Get the number of scheduled tokens for each request.
        use_max_model_len = self.most_model_len is None
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        end_index = start_index

        # Use either most_model_len or max_model_len depending on request size.
        for i in range(start_index, num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if not use_max_model_len and num_tokens > self.most_model_len:
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)
        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[
                    : self.num_reqs_max_model_len
                ]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[
                    : self.num_reqs_most_model_len
                ]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs
        max_num_scheduled_tokens_all_reqs = max(num_scheduled_tokens_per_req)
        num_scheduled_tokens_per_req = np.array(
            num_scheduled_tokens_per_req, dtype=np.int32
        )
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        assert max_num_scheduled_tokens_all_reqs > 0

        num_reqs = len(num_scheduled_tokens_per_req)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req]
        )

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.input_ids_cpu[:total_num_scheduled_tokens],
        )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(
            num_scheduled_tokens_per_req, out=self.query_start_loc_np[1 : num_reqs + 1]
        )
        self.query_start_loc_np[num_reqs + 1 :] = 1

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs]
            + num_scheduled_tokens_per_req
        )

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens
        )
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens
        ] = 0
        self.input_ids = self.input_ids_cpu[:padded_total_num_scheduled_tokens].to(
            self.device
        )
        self.position_ids = self.positions_cpu[:padded_total_num_scheduled_tokens].to(
            self.device
        )
        if use_max_model_len:
            block_tables = self.block_table_cpu[
                : self.num_reqs_max_model_len, : self.max_num_blocks_per_req
            ]
            block_tables[:num_reqs, : self.max_num_blocks_per_req] = (
                self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs]
            )
            query_start_loc = self.query_start_loc_cpu[
                : self.num_reqs_max_model_len + 1
            ].to(self.device)
            seq_lens = self.seq_lens_cpu[: self.num_reqs_max_model_len].to(self.device)
        else:
            block_tables = self.block_table_cpu[
                : self.num_reqs_most_model_len, : self.num_blocks_per_most_len_req
            ]
            block_tables[:num_reqs, : self.num_blocks_per_most_len_req] = (
                self.input_batch.block_table[0].get_cpu_tensor()[
                    :num_reqs, : self.num_blocks_per_most_len_req
                ]
            )
            query_start_loc = self.query_start_loc_cpu[
                : self.num_reqs_most_model_len + 1
            ].to(self.device)
            seq_lens = self.seq_lens_cpu[: self.num_reqs_most_model_len].to(self.device)
        block_tables = block_tables.to(self.device)

        # Calculate the slot mapping
        slot_mapping_metadata = self._get_slot_mapping_metadata(
            num_reqs, num_scheduled_tokens_per_req
        )
        num_kv_update_slices = slot_mapping_metadata.shape[0]
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            padded_total_num_scheduled_tokens, self.max_num_reqs, self.block_size
        )
        slot_mapping_metadata = np.pad(
            slot_mapping_metadata,
            [[0, padded_num_slices - len(slot_mapping_metadata)], [0, 0]],
            constant_values=0,
        )
        slot_mapping_metadata = np.transpose(slot_mapping_metadata)
        slot_mapping_metadata = torch.tensor(slot_mapping_metadata, device=self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens
            )

            self.set_active_loras(self.input_batch, padded_num_scheduled_tokens_per_req)

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping_metadata,
            block_tables=block_tables,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=torch.tensor([num_reqs], dtype=torch.int32, device=self.device),
            num_kv_update_slices=torch.tensor(
                [num_kv_update_slices], dtype=torch.int32, device=self.device
            ),
            num_slices_per_kv_cache_update_block=self._num_slices_per_kv_cache_update_block,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs
        )
        # Indices at which we sample (positions of last token in the sequence).
        # Padded to avoid recompiling when `num_reqs` varies.
        logits_indices = self.query_start_loc_cpu[1 : padded_num_reqs + 1] - 1
        logits_indices = logits_indices.to(self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens
            )

            self.set_active_loras(self.input_batch, padded_num_scheduled_tokens_per_req)

        layer_names = get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata for layer_name in layer_names
        }
        return (
            per_layer_attn_metadata,
            logits_indices,
            padded_num_reqs,
            num_reqs,
            end_index,
        )

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
            multimodal_cpu_fields=model.multimodal_cpu_fields,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            torch_xla.sync(wait=False)
            curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)
            torch_xla.sync(wait=False)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            if isinstance(curr_group_outputs, torch.Tensor):
                encoder_outputs.append(curr_group_outputs)
            else:
                assert isinstance(curr_group_outputs, (list, tuple))
                for output in curr_group_outputs:
                    encoder_outputs.append(output)

        # Cache the encoder outputs.
        # NOTE (NickLucche) here we diverge from logic in other runners, as we
        # assume to only have whole mm items to process. Hence we avoid the
        # intrinsic dynamism that `scatter_mm_placeholders` introduces.
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            assert pos_info.is_embed is None, (
                "Expected all positions to be contiguous and embeddings."
            )
            self.encoder_cache[mm_hash] = output

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens
        )

        is_mm_embed = self.is_mm_embed_cpu
        is_mm_embed[:padded_total_num_scheduled_tokens] = False
        mm_embeds = list[torch.Tensor]()
        req_start_idx = 0

        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens

            # TODO unroll loop and assume/enforce --disable_chunked_mm_input
            # NOTE (NickLucche) here we diverge from logic in other runners, as
            # we assume to only have whole mm items to process. Hence we avoid
            # the intrinsic dynamism that `gather_mm_placeholders` introduces.
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
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
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                assert pos_info.is_embed is None, (
                    "Expected all positions to be contiguous and embeddings."
                )

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = True

                # Only whole mm items are processed
                mm_embeds.append(encoder_output)

            req_start_idx += num_scheduled_tokens

        is_mm_embed = is_mm_embed[:padded_total_num_scheduled_tokens].to(self.device)

        return mm_embeds, is_mm_embed

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ):
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | None:
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        mm_embed_inputs = None
        if self.supports_mm_inputs:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embed_inputs = self._gather_mm_embeddings(scheduler_output)

        torch_xla.sync(wait=False)

        self.scheduler_output = scheduler_output
        self.mm_embed_inputs = mm_embed_inputs
        return None

    @torch.no_grad()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput:
        if self.scheduler_output is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            return None  # noqa
        scheduler_output = self.scheduler_output
        mm_embed_inputs = self.mm_embed_inputs
        self.scheduler_output = None
        self.mm_embed_inputs = None

        # Prepare inputs, the requests might be split into multiple
        # executions, combine the result of each execution.
        start_index = 0
        combined_selected_tokens: list[torch.Tensor] = []
        combined_logprobs: list[LogprobsLists] = []

        # NOTE: setup current batch's metadata for kv connector.
        # Currently, only verified with NixlConnector
        with set_forward_context(None, self.vllm_config):
            self.maybe_setup_kv_connector(scheduler_output)

        while start_index < self.input_batch.num_reqs:
            attn_metadata, logits_indices, padded_num_reqs, num_reqs, end_index = (
                self._prepare_inputs(scheduler_output, start_index)
            )
            input_ids, inputs_embeds = self._get_model_inputs(
                self.input_ids, mm_embed_inputs
            )
            torch_xla.sync(wait=False)
            # Run the decoder
            with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self.position_ids,
                    inputs_embeds=inputs_embeds,
                )
            hidden_states = self.select_hidden_states(hidden_states, logits_indices)
            logits = self.compute_logits(hidden_states)
            tpu_sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
                self.input_batch, padded_num_reqs, self.device
            )
            if grammar_output is not None:
                require_struct_decoding, grammar_bitmask_padded, arange = (
                    self.prepare_structured_decoding_input(logits, grammar_output)
                )
                logits = self.structured_decode(
                    require_struct_decoding, grammar_bitmask_padded, logits, arange
                )
            selected_token_ids = self.sample_from_logits_func(
                logits, tpu_sampling_metadata
            )
            # NOTE (NickLucche) Use the original logits (before any penalties or
            # temperature scaling) for the top-k logprobs. We can't enforce it
            # due to recompilations outside torch.compiled code, so just make
            # sure `sample_from_logits` does not modify the logits in-place.
            logprobs = (
                self.gather_logprobs(logits, selected_token_ids)
                if tpu_sampling_metadata.logprobs
                else None
            )

            # Remove padding on cpu and keep dynamic op outside of xla graph.
            selected_token_ids = selected_token_ids.cpu()[:num_reqs]

            combined_selected_tokens.append(selected_token_ids)
            if tpu_sampling_metadata.logprobs:
                combined_logprobs.append(logprobs.tolists())

            start_index = end_index

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        self.maybe_wait_for_kv_save()
        finished_sending, finished_recving = self.get_finished_kv_transfers(
            scheduler_output
        )

        selected_token_ids = torch.cat(combined_selected_tokens, dim=0)
        if tpu_sampling_metadata.logprobs:

            def concat_lists(input_lists):
                result = []
                for input_list in input_lists:
                    result.extend(input_list)
                return result

            logprobs_lists = LogprobsLists(
                logprob_token_ids=concat_lists(
                    [lp.logprob_token_ids for lp in combined_logprobs]
                ),
                logprobs=concat_lists([lp.logprobs for lp in combined_logprobs]),
                sampled_token_ranks=concat_lists(
                    [lp.sampled_token_ranks for lp in combined_logprobs]
                ),
            )
        else:
            logprobs_lists = None

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in self.input_batch.req_ids[:num_reqs]
        ), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()

            # Mask out the sampled tokens that should not be sampled.
            # TODO: Keep in sync with gpu_model_runner.py, in particular
            #       the "else" case here
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Append sampled tokens
            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
                self.input_batch.num_tokens[i] += 1

        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist() for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            self.input_batch.num_tokens[:num_reqs] += gen_lens
            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.input_batch.token_ids_cpu[i, target_slice] = (
                    valid_sampled_token_ids[i]
                )
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        kv_connector_output = (
            None
            if (finished_sending is None and finished_recving is None)
            else KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            )
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )

        # Check there are no new graphs compiled - all the graphs should be
        # captured and compiled during warm up.
        self._verify_num_xla_graphs("execute_model")

        return model_runner_output

    def update_config(self, overrides: dict[str, Any]) -> None:
        # TODO: TPU config may need extra validation
        # https://github.com/vllm-project/vllm/pull/20095#discussion_r2201497754
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.
        xm_tp_rank = xr.global_ordinal()
        with patch(
            "vllm.model_executor.layers.vocab_parallel_embedding."
            "get_tensor_model_parallel_rank",
            return_value=xm_tp_rank,
        ):
            try:
                if self.use_spmd:
                    tpu_loader = TPUModelLoader(
                        load_config=self.vllm_config.load_config
                    )
                    model = tpu_loader.load_model(
                        vllm_config=self.vllm_config,
                        model_config=self.vllm_config.model_config,
                        mesh=self.mesh,
                    )
                else:
                    model_loader = get_model_loader(self.load_config)
                    logger.info("Loading model from scratch...")
                    model = model_loader.load_model(
                        vllm_config=self.vllm_config, model_config=self.model_config
                    )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Unable to load model, a likely reason is the model is "
                    "too large for the current device's HBM memory. "
                    "Consider switching to a smaller model "
                    "or sharding the weights on more chips. "
                    f"See the detailed error: {e}"
                ) from e
        if self.lora_config is not None:
            model = self.load_lora_model(model, self.vllm_config, self.device)
            replace_set_lora(model)

        # Sync all pending XLA execution during model initialization and weight
        # loading.
        torch_xla.sync(wait=False)
        xm.wait_device_ops()
        if not hasattr(self, "model"):
            self.model = model
        self.sampler = TPUSampler()

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."
        )
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.model, model_config=self.model_config)

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int, num_reqs: int, num_blocks: int) -> None:
        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = torch.zeros(
                (num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
            )
        else:
            input_ids = torch.zeros((num_tokens), dtype=torch.int32).to(self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, num_reqs)
        position_ids = torch.zeros(num_tokens, dtype=torch.int32).to(self.device)
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
            num_tokens, self.max_num_reqs, self.block_size
        )
        num_kv_update_slices = torch.tensor([padded_num_slices], dtype=torch.int32).to(
            self.device
        )
        slot_mapping = torch.zeros((3, padded_num_slices), dtype=torch.int32).to(
            self.device
        )
        block_tables = torch.zeros((num_reqs, num_blocks), dtype=torch.int32).to(
            self.device
        )
        query_lens = [1] * num_reqs
        query_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int32), dim=0, dtype=torch.int32
        ).to(self.device)
        context_lens = torch.ones((num_reqs,), dtype=torch.int32).to(self.device)
        num_seqs = torch.tensor([actual_num_reqs], dtype=torch.int32).to(self.device)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            num_kv_update_slices=num_kv_update_slices,
            num_slices_per_kv_cache_update_block=self._num_slices_per_kv_cache_update_block,
        )

        if self.supports_mm_inputs:
            torch._dynamo.mark_dynamic(inputs_embeds, 0)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
        torch._dynamo.mark_dynamic(attn_metadata.block_tables, (0, 1))
        torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
        torch._dynamo.mark_dynamic(attn_metadata.query_start_loc, 0)

        layer_names = get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata for layer_name in layer_names
        }

        with (
            self.maybe_select_dummy_loras(
                self.lora_config, np.array([num_tokens], dtype=np.int32)
            ),
            set_forward_context(per_layer_attn_metadata, self.vllm_config, 0),
        ):
            out = self.model(
                input_ids=input_ids, positions=position_ids, inputs_embeds=inputs_embeds
            )
        self._hidden_states_dtype = out.dtype

    def _set_active_loras(
        self, prompt_lora_mapping, token_lora_mapping, lora_requests
    ) -> None:
        torch_xla.sync(wait=False)  # Captures input updates
        super()._set_active_loras(
            prompt_lora_mapping, token_lora_mapping, lora_requests
        )
        torch_xla.sync(wait=False)  # Captures metadata updates

    def _precompile_mm_encoder(self) -> None:
        if not self.supports_mm_inputs:
            return

        # Pre-compile MM encoder for all supported data modalities.
        hf_config = self.vllm_config.model_config.hf_config

        mm_budget = self.mm_budget
        assert mm_budget is not None

        max_items_per_seq_by_modality = mm_budget.max_items_per_batch_by_modality  # noqa: E501

        for mode, max_items_per_seq in max_items_per_seq_by_modality.items():
            logger.info(
                "Compiling Multimodal %s Encoder with different input shapes.", mode
            )
            start = time.perf_counter()
            # No padding for MM encoder just yet.
            for num_items in range(1, max_items_per_seq + 1):
                logger.info("  -- mode: %s items: %d", mode, num_items)
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    mode,
                    num_items,
                )
                # Run multimodal encoder.
                torch_xla.sync(wait=False)
                mm_embeds = self.model.embed_multimodal(**batched_dummy_mm_inputs)
                torch_xla.sync(wait=False)
                num_patches = mm_embeds[0].shape[0]
                items_size = num_patches * num_items

                # NOTE (NickLucche) pre-compile `embed_input_ids` when mm
                # embeddings are present. We assume `--disable-mm-chunked`,
                # hence only whole items can be scheduled. This implies we just
                # need to compile when `num_items` fit the (padded) `input_ids`
                for num_tokens in self.num_tokens_paddings:
                    if num_tokens >= items_size:
                        # XLA Workaround: if torch.zeros(..device) is used, XLA
                        # compiles a scalar+expansion op, which won't match
                        # the graph generated at runtime. CPU->TPU must be used
                        placeholders_ids = torch.zeros(
                            num_tokens, dtype=torch.int32, device="cpu"
                        )
                        # Align placeholders and actual num mm_embeddings.
                        placeholders_ids[:items_size] = hf_config.image_token_index

                        placeholders_ids = placeholders_ids.to(self.device)

                        mm_mask = torch.tensor([False] * num_tokens)
                        mm_mask[:items_size] = True
                        mm_mask = mm_mask.to(self.device)
                        # Assign outputs or the graph will be cut short.
                        a, b = self._get_model_inputs(
                            placeholders_ids,
                            mm_embed_inputs=([mm_embeds], mm_mask),
                        )
                        assert a is None
                        torch_xla.sync(wait=False)

            # Pre-compile `embed_input_ids` when mm_embeddings are not
            # present. Chunk is only made of text, no mm_placeholders.
            for num_tokens in self.num_tokens_paddings:
                placeholders_ids = torch.zeros(
                    num_tokens, dtype=torch.int32, device="cpu"
                )
                placeholders_ids = placeholders_ids.to(self.device)
                a, b = self._get_model_inputs(
                    placeholders_ids,
                    mm_embed_inputs=None,
                )
                assert a is None
                torch_xla.sync(wait=False)

            xm.wait_device_ops()
            end = time.perf_counter()
            logger.info(
                "Multimodal %s Encoder compilation finished in in %.2f [secs].",
                mode,
                end - start,
            )

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(
                num_tokens, self.num_reqs_max_model_len, self.max_num_blocks_per_req
            )
            if self.most_model_len is not None:
                self._dummy_run(
                    num_tokens,
                    self.num_reqs_most_model_len,
                    self.num_blocks_per_most_len_req,
                )
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("model backbone")

    def _precompile_select_hidden_states(self) -> None:
        # Compile hidden state selection function for bucketed
        # n_tokens x max_num_reqs. Graph is really small so this is fine.
        logger.info("Compiling select_hidden_states with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            dummy_hidden = torch.zeros(
                (num_tokens, hsize), device=self.device, dtype=self._hidden_states_dtype
            )
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            for num_reqs in self.num_reqs_paddings:
                indices = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
                torch._dynamo.mark_dynamic(indices, 0)
                self.select_hidden_states(dummy_hidden, indices)
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens, num_reqs)
                # Requests can't be more than tokens. But do compile for the
                # next bigger value in case num_tokens uses bucketed padding.
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("select_hidden_states")

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_reqs in self.num_reqs_paddings:
            dummy_hidden = torch.zeros(
                (num_reqs, hsize), device=self.device, dtype=self._hidden_states_dtype
            )
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            self.compute_logits(dummy_hidden)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("compute_logits")

    def _precompile_structured_decoding(self) -> None:
        logger.info("Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_require_struct_decoding = self.require_structured_out_cpu[
                :num_reqs
            ].to(self.device)
            dummy_grammar_bitmask = self.grammar_bitmask_cpu[:num_reqs].to(self.device)
            # The first dimension of the above 3 dummy tensors cannot be
            # mark_dynamic because some operations in structured_decode require
            # them to be static.
            arange = self.structured_decode_arange.to(self.device)
            self.structured_decode(
                dummy_require_struct_decoding,
                dummy_grammar_bitmask,
                dummy_logits,
                arange,
            )
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("structured_decoding")

    def _precompile_sample_from_logits(self) -> None:
        logger.info("Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            # The first dimension of dummy_logits cannot be mark_dynamic
            # because some operations in the sampler require it to be static.
            for all_greedy in [False, True]:
                generate_params_if_all_greedy = not all_greedy
                sampling_metadata = TPUSupportedSamplingMetadata.from_input_batch(
                    self.input_batch,
                    num_reqs,
                    self.device,
                    generate_params_if_all_greedy,
                )
                sampling_metadata.all_greedy = all_greedy
                with self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_reqs], dtype=np.int32)
                ):
                    self.sample_from_logits_func(dummy_logits, sampling_metadata)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("sample_from_logits")

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros(
                (num_reqs, self.vocab_size),
                device=self.device,
                dtype=self._hidden_states_dtype,
            )
            dummy_tokens = torch.zeros((num_reqs, 1), dtype=torch.int64).to(self.device)
            with self.maybe_select_dummy_loras(
                self.lora_config, np.array([num_reqs], dtype=np.int32)
            ):
                self.gather_logprobs(dummy_logits, dummy_tokens)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("gather_logprobs")

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        with self.maybe_setup_dummy_loras(self.lora_config):
            self._precompile_mm_encoder()
            self._precompile_backbone()
            self._precompile_select_hidden_states()
            self._precompile_compute_logits()
            self._precompile_structured_decoding()
            self._precompile_sample_from_logits()
            self._precompile_gather_logprobs()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            if self.model_config.multimodal_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache."
                )
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                # TODO: handle encoder-decoder models once we support them.
                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[
                        dummy_modality
                    ]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    # Isolate encoder graph from post-processing to minimize
                    # impact of recompilation until it's fixed.
                    start = time.perf_counter()
                    torch_xla.sync(wait=False)
                    dummy_encoder_outputs = self.model.embed_multimodal(
                        **batched_dummy_mm_inputs
                    )
                    torch_xla.sync(wait=False)
                    xm.wait_device_ops()
                    end = time.perf_counter()
                    logger.info(
                        "Multimodal Encoder profiling finished in %.2f [secs].",
                        end - start,
                    )

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )

                    # Cache the dummy encoder outputs.
                    self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Trigger compilation for general shape.
        self._dummy_run(
            num_tokens, self.num_reqs_max_model_len, self.max_num_blocks_per_req
        )
        if self.most_model_len is not None:
            self._dummy_run(
                num_tokens,
                self.num_reqs_most_model_len,
                self.num_blocks_per_most_len_req,
            )

        torch_xla.sync(wait=False)
        xm.wait_device_ops()
        self.encoder_cache.clear()
        gc.collect()

    def maybe_setup_cross_layer_kv_sharing(
        self,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
        )

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not supported yet."
            )

        if (
            kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            != self.block_size
        ):
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
                kernel_block_sizes=[
                    kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
                ],
            )
        # Verify dtype compatibility between block_table_cpu and input_batch
        assert (
            self.block_table_cpu.dtype
            == self.input_batch.block_table[0].get_cpu_tensor().dtype
        )

        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache tensor shared by multiple layers is not supported in TPU."
            )
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        kv_caches: dict[str, torch.Tensor] = {}
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_size = kv_cache_sizes[layer_name]
                assert tensor_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_size // kv_cache_spec.page_size_bytes  # noqa
                if isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        assert self.original_parallel_config is not None
                        tp_size = self.original_parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode"
                        )
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape, dtype=dtype).to(
                        self.device
                    )

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        # Set up cross-layer KV cache sharing if needed
        self.maybe_setup_cross_layer_kv_sharing(kv_caches, kv_cache_config)

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if self.use_spmd:
            # Shard KV Cache
            for cache in self.kv_caches:
                xs.mark_sharding(cache, self.mesh, (None, "x", None, None))

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)

    def reset_dynamo_cache(self):
        # NOTE: We check `is_multimodal_model` instead of `supports_mm_inputs`
        # since the compiled model object of the language backbone of a
        # multimodal model needs to be extracted via `get_language_model`.
        if self.model_config.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWrapperWithCustomDispatcher):
            logger.info("Clear dynamo cache and cached dynamo bytecode.")
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object
            )
            compiled_model.compiled_codes.clear()

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def select_hidden_states(self, hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def compute_logits(self, sample_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(sample_hidden_states)

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    # @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def sample_from_logits(
        self, logits: torch.Tensor, sampling_metadata: TPUSupportedSamplingMetadata
    ) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if sampling_metadata.all_greedy:
            out_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            out_tokens = self.sampler(logits, sampling_metadata).sampled_token_ids
        return out_tokens

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def gather_logprobs(
        self, logits: torch.Tensor, sampled_tokens: torch.Tensor
    ) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        logprobs = self.sampler.compute_logprobs(logits)
        return self.sampler.gather_logprobs(
            logprobs,
            self.model_config.max_logprobs,
            token_ids=sampled_tokens.squeeze(-1),
        )

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits,
        )

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor, arange: torch.Tensor
    ):
        assert logits.shape[0] == grammar_bitmask.shape[0]
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (
                torch.bitwise_right_shift(grammar_bitmask[i][:, None], arange[None, :])
                & 1
            ) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[: self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf")
            )
        return logits_cloned

    def embed_multimodal(self, *args, **kwargs):
        return self.model.embed_multimodal(*args, **kwargs)

    def embed_input_ids(self, *args, **kwargs):
        return self.model.embed_input_ids(*args, **kwargs)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, grammar_output: "GrammarOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = grammar_output.grammar_bitmask
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        cumulative_mask_idx = 0
        for req_id in grammar_output.structured_output_request_ids:
            if req_id not in self.input_batch.req_id_to_index:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            self.grammar_bitmask_cpu[batch_index] = torch.from_numpy(
                grammar_bitmask[cumulative_mask_idx]
            )
            # It's not guaranteed that all requests in this batch require
            # structured output, so create a bool tensor to represent
            # the requests that need structured output.
            self.require_structured_out_cpu[batch_index] = True
            cumulative_mask_idx += 1

        return (
            self.require_structured_out_cpu[:num_reqs].to(logits.device),
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
        )

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch

        model = cast(SupportsMultiModal, self.model)
        return next(
            grouped_mm_kwargs
            for _, _, grouped_mm_kwargs in group_mm_kwargs_by_modality(
                dummy_mm_items,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
                multimodal_cpu_fields=model.multimodal_cpu_fields,
            )
        )


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_token_paddings(
    min_token_size: int, max_token_size: int, padding_gap: int
) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _get_padded_num_kv_cache_update_slices(
    num_tokens: int, max_num_reqs: int, page_size: int
) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    # NOTE(chengjiyao): let's say R_i is the token num for i-th request,
    # so it occupies most 2 + R_i // page_size pages. The total maximum
    # possible number of pages needed is sum(2 + R_i // page_size), which
    # is <= 2 * max_num_reqs + sum(R_i) // page_size
    # = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices


def _get_num_slices_per_kv_cache_update_block(page_size_bytes: int) -> int:
    """Find the optimum number of slices to copy per Pallas program instance.

    Increasing the number of slices copied in one instance of the kernel program
    will increase HBM bandwidth utilization via more in-flight DMAs.

    However, it will also use more VMEM, and experimentally, we observed
    performance regression at 128 slices on v6e, likely due to running
    out of scalar registers. Thus this function will limit the number of
    slices to 64.
    """
    # The default vmem_limit_bytes of a pallas kernel is 32MB. Here we
    # calculate num_slices_per_block based on 16MB in case any register spills.
    vmem_limit = 16 * 1024 * 1024
    num_slices_per_block = vmem_limit // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = prev_power_of_2(num_slices_per_block)
    if num_slices_per_block > 64:
        num_slices_per_block = 64
    return num_slices_per_block


def replace_set_lora(model):
    def _tpu_set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: torch.Tensor | None,
    ):
        # TODO: The integer index leads to a recompilation, but converting it
        # to a tensor doesn't seem to work anymore. This might be fixed with a
        # later release of torch_xla.
        self._original_set_lora(index, lora_a, lora_b, embeddings_tensor)
        torch_xla.sync(wait=False)

    def _tpu_reset_lora(self, index: int):
        self._original_reset_lora(index)
        torch_xla.sync(wait=False)

    for _, module in model.named_modules():
        if isinstance(module, BaseLayerWithLoRA):
            module._original_set_lora = module.set_lora
            module._original_reset_lora = module.reset_lora
            module.set_lora = _tpu_set_lora.__get__(module, module.__class__)
            module.reset_lora = _tpu_reset_lora.__get__(module, module.__class__)
