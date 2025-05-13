# SPDX-License-Identifier: Apache-2.0
from typing import Any

import jax
import jax.numpy as jnp

from vllm.v1.worker.input_batch_jax import InputBatch as InputBatchJax
from vllm.v1.worker.tpu_model_runner import *
from vllm.v1.worker.tpu_model_runner import (_get_req_paddings,
                                             _get_token_paddings)

logger = init_logger(__name__)


class TPUModelRunnerJax:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: Any,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self.kv_caches: list[jax.Array] = []

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self._hidden_states_dtype = self.dtype

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.vocab_size = model_config.get_vocab_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, jax.Array]] = {}

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatchJax(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.vocab_size,
        )

        self.input_ids = jnp.zeros((self.max_num_tokens, ), dtype=jnp.int32)

        self.positions = jnp.zeros((self.max_num_tokens, ), dtype=jnp.int32)

        # Assuming self.input_batch.block_table is already a JAX array
        # Access its dtype directly
        block_table_dtype = self.input_batch.block_table.dtype
        self.block_table = jnp.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=block_table_dtype)

        self.query_start_loc = jnp.zeros((self.max_num_tokens + 1, ),
                                         dtype=jnp.int32)

        self.seq_lens = jnp.zeros((self.max_num_tokens, ), dtype=jnp.int32)

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

    def _update_num_xla_graphs(self, case_str):
        return

    def _verify_num_xla_graphs(self, case_str):
        return

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

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
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
                req_data.num_computed_tokens)
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
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

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def get_model(self):
        assert self.model is not None
        return self.model

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        model_config = self.vllm_config.model_config
        parallel_config = self.vllm_config.parallel_config
        for i in range(model_config.get_num_layers(parallel_config)):
            kv_cache_spec[f"layers.{i}"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=model_config.get_num_attention_heads(
                    parallel_config),
                head_size=model_config.get_head_size(),
                dtype=jnp.bfloat16,
                use_mla=False,
            )

        return kv_cache_spec

    def _get_model_inputs(self, input_ids: torch.Tensor,
                          mm_embeds: list[torch.Tensor]):
        return input_ids, None

    def load_model(self) -> None:
        self.device = self.device_config.device
        model = get_model(vllm_config=self.vllm_config)
        self.model = model

    def _dummy_run(self, num_tokens: int) -> None:
        if self.model is None:
            self.load_model()

        input_ids = jnp.zeros((num_tokens, ), dtype=jnp.int32)
        # device=device) # Explicit placement if needed

        inputs_embeds = None  # Remains None

        actual_num_reqs = min(num_tokens, self.max_num_reqs)

        position_ids = jnp.zeros((num_tokens, ), dtype=jnp.int32)
        # device=device) # Explicit placement if needed

        slot_mapping = jnp.zeros(
            (num_tokens, ), dtype=jnp.int64)  # Note: JAX default int is int32
        # device=device) # Explicit placement if needed

        block_tables = jnp.zeros(
            (self.max_num_reqs,
             self.block_table.shape[1]),  # Use shape directly
            dtype=jnp.int32)
        # device=device) # Explicit placement if needed

        query_lens = [1] * self.max_num_reqs  # Standard Python list

        # JAX equivalent for cumsum and tensor creation
        # Create the initial array, potentially on the device
        initial_query_loc = jnp.array([0] + query_lens, dtype=jnp.int32)
        # device=device) # Explicit placement if needed

        # Perform cumsum. The result typically stays on the same device.
        query_start_loc = jnp.cumsum(initial_query_loc,
                                     axis=0,
                                     dtype=jnp.int32)

        context_lens = jnp.ones((self.max_num_reqs, ), dtype=jnp.int32)
        # device=device) # Explicit placement if needed

        num_seqs = jnp.array([actual_num_reqs], dtype=jnp.int32)
        # device=device) # Explicit placement if needed
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
        )

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }

        out = self.model(input_ids=input_ids,
                         positions=position_ids,
                         inputs_embeds=inputs_embeds)

        return out

    def profile_run(self, num_tokens: int):
        self._dummy_run(num_tokens)

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(num_tokens)
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        self._precompile_backbone()
        print(">>>>> Ready to be submitted <<<<<")
        assert False

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = jnp.zeros(kv_cache_shape, dtype=dtype)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)
