# SPDX-License-Identifier: Apache-2.0

import gc
import time
from typing import List

import numpy as np
import torch
import torch.distributed

from vllm.config import CompilationLevel
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.utils import DeviceMemoryProfiler, cdiv
from vllm.v1.worker.model_runner_base import ModelRunnerBase

logger = init_logger(__name__)


class GPUModelRunner(ModelRunnerBase):

    def _cache_device_properties(self):
        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # Cache the device properties.
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def profile_run(self) -> None:
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

        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            max_tokens_by_modality_dict = MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(  # noqa: E501
                self.model_config)
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
            dummy_request_data = self.input_registry.dummy_data_for_profiling(
                model_config=self.model_config,
                seq_len=self.max_num_tokens,
                mm_registry=self.mm_registry,
            )
            dummy_mm_data = dummy_request_data.multi_modal_data

            # Dummy data definition in V0 may contain multiple multimodal items
            # (e.g, multiple images) for a single request, therefore here we
            # always replicate first item by max_num_mm_items times since in V1
            # they are scheduled to be processed separately.

            # Case when models have a merged processor, their dummy data is
            # already batched `MultiModalKwargs`, therefore we take the first
            # `MultiModalKwargsItem` from the desired modality to profile on.
            if isinstance(dummy_mm_data, MultiModalKwargs):
                dummy_mm_item = dummy_mm_data.get_item(
                    modality=dummy_data_modality, item_index=0)
                dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

            # Case when models have dummy data explicitly defined as
            # `MultiModalDataDict`, so they need to be processed through input
            # mapper.
            # TODO (ywang96): deprecate this path once merged processor is
            # supported on all models.
            else:
                mm_kwargs_list = self.mm_input_mapper_profiling.process_inputs(
                    mm_data=dummy_mm_data,
                    mm_hashes=None,
                    mm_processor_kwargs=None,
                    precomputed_mm_inputs=None)
                dummy_mm_kwargs = mm_kwargs_list[0]

            batched_dummy_mm_inputs = MultiModalKwargs.batch(
                [dummy_mm_kwargs] * max_num_mm_items)
            batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
                batched_dummy_mm_inputs, device=self.device)

            # Run multimodal encoder.
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)
            assert len(dummy_encoder_outputs) == max_num_mm_items, (
                "Expected dimension 0 of encoder outputs to match the number "
                f"of multimodal data items: {max_num_mm_items}, got "
                f"{len(dummy_encoder_outputs)=} instead. This is most likely "
                "due to the 'get_multimodal_embeddings' method of the model "
                "not implemented correctly.")

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        num_reqs = self.scheduler_config.max_num_seqs
        num_tokens = self.max_num_tokens
        min_tokens_per_req: int = num_tokens // num_reqs

        num_scheduled_tokens_list: List[int] = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        num_scheduled_tokens: np.ndarray = np.array(num_scheduled_tokens_list,
                                                    dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1

        with self.maybe_profile_with_lora(self.lora_config,
                                          num_scheduled_tokens):
            # Trigger compilation for general shape.
            hidden_states = self._dummy_run(self.max_num_tokens,
                                            dummy_kv_caches)
            hidden_states = hidden_states[logit_indices]
            logits = self.model.compute_logits(hidden_states, None)
            # TODO(woosuk): Consider the memory usage of the sampler.
            torch.cuda.synchronize()
            del hidden_states, logits
            self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> None:
        if not self.use_cuda_graph:
            logger.warning(
                "Skipping CUDA graph capture. Please add "
                "-O %s to use CUDA graphs.", CompilationLevel.PIECEWISE)
            return

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens)
                self._dummy_run(num_tokens)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
