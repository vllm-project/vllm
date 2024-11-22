import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
import pickle as pkl

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm import envs
from vllm.compilation.compile_context import set_compile_context
from vllm.compilation.config import CompilationConfig
from vllm.compilation.levels import CompilationLevel
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs
from vllm.plugins import set_compilation_config
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler, cdiv,
                        is_pin_memory_available)
from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend,
                                                   FlashAttentionMetadata)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.cached_request_state import CachedRequestState
from vllm.v1.worker.request_batch import RequestBatch
from vllm.v1.worker.model_runner_device_tensors import ModelRunnerDeviceTensors
from vllm.v1.worker.lora_request_batch import LoRARequestBatch

from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

if TYPE_CHECKING:
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):

    STEP = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        input_registry: InputRegistry = INPUT_REGISTRY,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        self.input_registry = input_registry

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch. All tensors maintained by the batch are on the CPU.
        self.request_batch = LoRARequestBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            pin_memory=self.pin_memory,
        )

        # Device Tensors
        self.device_tensors = ModelRunnerDeviceTensors.make(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_num_tokens = self.max_num_tokens,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            input_embeds_hidden_size=self.hidden_size,
            input_embeds_dtype=self.dtype,
            device=self.device)

        self.use_cuda_graph = (envs.VLLM_TORCH_COMPILE_LEVEL
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        self.cudagraph_batch_sizes = [1, 2, 4] + [i for i in range(8, 513, 8)]

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        self.request_batch.remove_requests(stopped_req_ids)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Grab num_existing_block_ids from request state before
            # update to request_state
            num_existing_block_ids = len(req_state.block_ids) 

            req_state.update(req_data)
            self.request_batch.update_states(req_id, req_data, num_existing_block_ids)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            sampling_params = req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
                prompt=req_data.prompt,
                mm_inputs=req_data.mm_inputs,
                mm_positions=req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request = req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.request_batch.add_request(req_state)

        # Condense the batched states if there are empty indices.
        self.request_batch.condense()

    def _prepare_inputs(self, num_scheduled_tokens: np.array, 
                        total_num_scheduled_tokens: Optional[int] = None) \
                            -> Tuple[torch.Tensor, FlashAttentionMetadata, torch.Tensor]:
        """
        Prepare model inputs such as, input_token_ids, attention metadata etc.
        This function triggers async CPU-GPU transfers some device tensors such
        as block_table, positions and slot_mapping.

        Args:
            num_scheduled_tokens (np.array): Numpy array containing the number of tokens
            scheduled to be processed for every request in self.request_batch. 
            Note that num_scheduled_tokens[i] must corresponding to the ith request in
            the request batch.

            total_num_scheduled_tokens (Optional[int]): Total number of tokens
            scheduled to be computed. This must be equal to np.sum(num_scheduled_tokens).
            This is an optional parameter for optimization.

        Returns:
            input_token_ids: Token ids from the scheduled requests. input_token_ids contains
                as many values as total_num_scheduled_tokens
            attention_metadata: FlashAttentionMetadata
            logits_indices: logits indices for model output readout.
        """
            
        if not total_num_scheduled_tokens:
            total_num_scheduled_tokens = np.sum(num_scheduled_tokens)

        assert total_num_scheduled_tokens > 0
        num_reqs: int = self.request_batch.num_reqs()
        assert num_reqs > 0

        # Prepare gpu tensors
        self.request_batch.prepare_inputs(num_scheduled_tokens, self.block_size,
                                        block_table_device_tensor=self.device_tensors.block_table,
                                        input_tokens_device_tensor=self.device_tensors.input_tokens,
                                        input_positions_device_tensor=self.device_tensors.input_positions,
                                        slot_mapping_device_tensor=self.device_tensors.slot_mapping)
        
        ## Prepare attention meta
        seq_lens_np: np.array = self.request_batch.make_seq_lens_tensor(num_scheduled_tokens)

        ## Query start loc
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        # make a numpy array
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        ## Seq start loc
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc_np = seq_start_loc.numpy()
        seq_start_loc_np[0] = 0
        np.cumsum(seq_lens_np, out=seq_start_loc_np[1:])

        max_seq_len = np.max(seq_lens_np)
        max_num_scheduled_tokens = np.max(num_scheduled_tokens) 

        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.device_tensors.block_table[:num_reqs],
            slot_mapping=self.device_tensors.slot_mapping[:total_num_scheduled_tokens],
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return self.device_tensors.input_tokens[:total_num_scheduled_tokens], attn_metadata, logits_indices

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        skip_copy = True
        if (scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        sampling_metadata = self.request_batch.make_sampling_metadata(self.device_tensors.sampling_tensors, skip_copy)
        return sampling_metadata

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: List[MultiModalKwargs] = []
        req_input_ids: List[Tuple[int, int]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[input_id])
                req_input_ids.append((req_id, input_id))
        batched_mm_inputs = MultiModalKwargs.batch(mm_inputs)
        batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                       device=self.device)

        # Run the encoder.
        # `encoder_outputs` is either of the following:
        # 1. A tensor of shape [num_images, feature_size, hidden_size]
        # in case when feature_size is fixed across all images.
        # 2. A list (length: num_images) of tensors, each of shape
        # [feature_size, hidden_size] in case when the feature size is
        # dynamic depending on input images.
        encoder_outputs = self.model.process_mm_inputs(**batched_mm_inputs)

        # Cache the encoder outputs.
        for (req_id, input_id), output in zip(req_input_ids, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = output

    def _gather_encoder_outputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> List[torch.Tensor]:
        encoder_outputs: List[torch.Tensor] = []
        for req_id in self.request_batch.request_ids():
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info["offset"]
                num_encoder_tokens = pos_info["length"]

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
                encoder_outputs.append(encoder_output[start_idx:end_idx])
        return encoder_outputs

    def dump_data(self, positions: torch.Tensor, input_ids: torch.Tensor, attn_metadata: FlashAttentionMetadata):

        print ("data dump : \n")
        print (f"   - input ids : {input_ids.shape} {input_ids.dtype} {input_ids}")
        print (f"   - positions : {positions.shape} {positions.dtype} {positions}")
        print (f"   - Flashattn : \n")
        print (f"       - num_actual_tokens : {attn_metadata.num_actual_tokens}")
        print (f"       - max_query_len     : {attn_metadata.max_query_len}")
        print (f"       - query_start_loc   : {attn_metadata.query_start_loc}")
        print (f"       - max_seq_len       : {attn_metadata.max_seq_len}")
        print (f"       - seq_start_loc     : {attn_metadata.seq_start_loc}")
        print (f"       - block_table       : {attn_metadata.block_table}")
        print (f"       - slot mapping      : {attn_metadata.slot_mapping}")
        return 

        # data tuple
        #data = (input_ids.cpu().numpy(), positions.cpu().numpy(),
        #                attn_metadata.num_actual_tokens,
        #                attn_metadata.max_query_len,
        #                attn_metadata.query_start_loc.cpu().numpy(),
        #                attn_metadata.max_seq_len,
        #                attn_metadata.seq_start_loc.cpu().numpy(),
        #                attn_metadata.block_table.cpu().numpy(),
        #                attn_metadata.slot_mapping.cpu().numpy(),
        #                self.input_batch.num_reqs)

        main_data = None
        fname = f"./dump/main_{self.STEP}.pkl"
        with open(fname, "rb") as f:
            main_data = pkl.load(f)

        main_input_ids, main_positions, main_actual_num_tokens, \
            main_max_query_len, main_query_start_loc, main_max_seq_len, \
                main_seq_start_loc, main_block_table, main_slot_mapping, main_num_reqs = main_data

        ## Tests
        assert main_num_reqs == self.request_batch.num_reqs()
        assert main_actual_num_tokens == attn_metadata.num_actual_tokens
        assert main_max_query_len == attn_metadata.max_query_len
        assert main_max_seq_len == attn_metadata.max_seq_len

        assert np.allclose(main_query_start_loc, attn_metadata.query_start_loc.cpu().numpy())
        assert np.allclose(main_seq_start_loc, attn_metadata.seq_start_loc.cpu().numpy())
        assert np.allclose(main_positions[:main_actual_num_tokens], positions.cpu().numpy()[:main_actual_num_tokens])
        assert np.allclose(main_input_ids, input_ids.cpu().numpy())
        assert np.allclose(main_slot_mapping, attn_metadata.slot_mapping.cpu().numpy())

        assert np.allclose(main_block_table[:main_num_reqs], attn_metadata.block_table.cpu().numpy()[:main_num_reqs])


        self.STEP += 1

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:

        self._update_states(scheduler_output)

        # Run the encoder.
        self._execute_encoder(scheduler_output)
        encoder_outputs = self._gather_encoder_outputs(scheduler_output)

        # Prepare the decoder inputs.
        num_scheduled_tokens: List[int] = []
        for req_id in self.request_batch.request_ids():
            num_scheduled_tokens.append(scheduler_output.num_scheduled_tokens[req_id])
        num_scheduled_tokens : np.array = np.array(num_scheduled_tokens)

        input_ids, attn_metadata, logits_indices = self._prepare_inputs(
            num_scheduled_tokens, scheduler_output.total_num_scheduled_tokens)

        # hot-swap lora model
        if self.lora_config:
            self.set_activte_loras(self.request_batch, num_scheduled_tokens)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self._get_padded_batch_size(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens

        # Get the inputs embeds.
        if encoder_outputs:
            inputs_embeds = self.model.get_input_embeddings(
                input_ids, encoder_outputs)
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
        # NOTE(woosuk): To unify token ids and soft tokens (vision embeddings),
        # always use embeddings (rather than token ids) as input to the model.
        # TODO(woosuk): Avoid the copy. Optimize.
        self.device_tensors.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)

        #self.dump_data(self.device_tensors.input_positions, input_ids, attn_metadata)

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata):
            hidden_states = self.model(
                input_ids=None,
                positions=self.device_tensors.input_positions[:num_input_tokens],
                kv_caches=self.kv_caches,
                attn_metadata=None,
                inputs_embeds=self.device_tensors.inputs_embeds[:num_input_tokens],
            )
        hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # NOTE: CPU-GPU synchronization happens here.
        sampled_token_ids = sampler_output.sampled_token_ids.cpu()
        sampled_token_ids_list = sampled_token_ids.tolist()
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        for i, req_id in enumerate(self.request_batch.request_ids()):
            # TODO (varun) : Move part of loop body into input_batch
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids_list[i]
                #self.input_batch.token_ids_cpu[i, seq_len] = token_id
                self.request_batch.append_token_id(req_id, token_id, seq_len)
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                self.request_batch.rewind_generator(req_id)
                #generator = self.input_batch.generators.get(i)
                #if generator is not None:
                #    # This relies on cuda-specific torch-internal impl details
                #    generator.set_offset(generator.get_offset() - 4)

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs.cpu()
        model_runner_output = ModelRunnerOutput(
            req_ids=self.request_batch.request_ids(),
            req_id_to_index={req_id : idx for idx, req_id in self.request_batch.request_ids()},
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    def load_model(self) -> None:
        if self.use_cuda_graph:
            # NOTE(woosuk): Currently, we use inductor because the piecewise
            # CUDA graphs do not work properly with the custom CUDA kernels.
            # FIXME(woosuk): Disable inductor to reduce the compilation time
            # and avoid any potential issues with the inductor.
            os.environ["VLLM_CUSTOM_OPS"] = "none"
            set_compilation_config(
                CompilationConfig(
                    use_cudagraph=True,
                    non_cudagraph_ops=["vllm.unified_v1_flash_attention"],
                    use_inductor=True,
                    enable_fusion=False,
                ))

        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model()

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def _dummy_run(self, model: nn.Module, num_tokens: int) -> None:
        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        dummy_kv_caches = [
            torch.tensor([], dtype=torch.float32, device=self.device)
            for _ in range(self.num_attn_layers)
        ]
        with set_forward_context(None):  # noqa: SIM117
            with set_compile_context(self.cudagraph_batch_sizes):
                # Trigger compilation for general shape.
                model(input_ids=None,
                      positions=self.device_tensors.input_positions,
                      kv_caches=dummy_kv_caches,
                      attn_metadata=None,
                      inputs_embeds=self.device_tensors.inputs_embeds)

    @torch.inference_mode()
    def profile_run(self) -> None:
        # TODO(woosuk): Profile the max memory usage of the encoder and
        # the encoder cache.
        self._dummy_run(self.model, self.max_num_tokens)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. Please set "
                "VLLM_TORCH_COMPILE_LEVEL=%d to use CUDA graphs.",
                CompilationLevel.PIECEWISE)
            return

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with set_forward_context(None):
            # Trigger CUDA graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                self.model(
                    input_ids=None,
                    positions=self.device_tensors.input_positions[:num_tokens],
                    kv_caches=self.kv_caches,
                    attn_metadata=None,
                    inputs_embeds=self.device_tensors.inputs_embeds[:num_tokens],
                )

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device))

    def _get_padded_batch_size(self, batch_size: int) -> Optional[int]:
        # TODO: Optimize this?
        for size in self.cudagraph_batch_sizes:
            if batch_size <= size:
                return size
        return None
