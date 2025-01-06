import time
from typing import Mapping, Optional, Union

from vllm.config import CacheConfig, LoRAConfig, ModelConfig
from vllm.inputs import (INPUT_REGISTRY, InputRegistry, ProcessorInputs,
                         PromptType, SingletonInputsAdapter)
from vllm.inputs.parse import is_encoder_decoder_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalHasher,
                             MultiModalKwargs, MultiModalRegistry)
from vllm.multimodal.utils import merge_and_sort_multimodal_metadata
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.mm_input_mapper import MMInputMapperClient


class Processor:

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        tokenizer: BaseTokenizerGroup,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        self.model_config = model_config
        self.lora_config = lora_config
        self.tokenizer = tokenizer

        self.generation_config_fields = model_config.try_get_generation_config(
        )
        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer,
                                                    mm_registry)
        self.input_processor = input_registry.create_input_processor(
            model_config)

        # Multi-modal (huggingface) input mapper
        self.mm_input_mapper_client = MMInputMapperClient(model_config)

        # Multi-modal hasher (for images)
        self.use_hash = (not model_config.disable_mm_preprocessor_cache) or \
            cache_config.enable_prefix_caching

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> EngineCoreRequest:

        # TODO(woosuk): Support pooling models.
        # TODO(woosuk): Check max_logprobs
        # TODO(woosuk): Support encoder-decoder models.

        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        assert priority == 0, "vLLM V1 does not support priority at the moment."
        assert trace_headers is None, "vLLM V1 does not support tracing yet."

        # Process inputs.
        preprocessed_inputs = self.input_preprocessor.preprocess(
            prompt,
            request_id=request_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)
        self._validate_model_inputs(processed_inputs)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

        if is_encoder_decoder_inputs(processed_inputs):
            decoder_inputs = SingletonInputsAdapter(
                processed_inputs["decoder"])
            encoder_inputs = SingletonInputsAdapter(
                processed_inputs["encoder"])
        else:
            decoder_inputs = SingletonInputsAdapter(processed_inputs)
            encoder_inputs = None

        # TODO: Impl encoder-decoder
        if encoder_inputs is not None:
            raise NotImplementedError

        assert isinstance(params, SamplingParams)
        # TODO: can we avoid cloning here in multiproc case
        sampling_params = params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)

        # Multimodal related.
        # Compute MM hashes (if enabled)
        mm_hashes = None
        if self.use_hash:
            # Use mm_hashes from processed inputs if the model has merged
            # input processor.
            if decoder_inputs.multi_modal_hashes:
                mm_hashes = decoder_inputs.multi_modal_hashes
            # Fallback to using MultiModalHasher directly.
            else:
                mm_hashes = MultiModalHasher.hash_prompt_mm_data(prompt)

        # For merged preprocessor, mm_data is already mm_inputs
        precomputed_mm_inputs: Optional[list[MultiModalKwargs]] = None
        decoder_mm_data = decoder_inputs.multi_modal_data
        if isinstance(decoder_mm_data, MultiModalKwargs):
            # The output of merged multi-modal processor (`decoder_mm_data`)
            # contains the kwargs for all items from all modalities.
            # This code separates them so that there is one set of kwargs
            # per item per modality.
            precomputed_mm_inputs = [
                MultiModalKwargs.from_items([item])
                for modality in decoder_mm_data.modalities
                for item in decoder_mm_data.get_items(modality)
            ]

        mm_positions = decoder_inputs.multi_modal_placeholders

        # Last-mile processing of multimodal metadata and inputs.
        if mm_positions:

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            # NOTE: interleaved modalities are not supported.
            (
                sorted_modalities,
                sorted_mm_positions,
                sorted_mm_hashes,
            ) = merge_and_sort_multimodal_metadata(
                mm_positions,
                mm_hashes,
            )

            # NOTE: Sort multimodal inputs/kwargs ONLY IF there are multiple
            # modalities involved AND the model supports merged input processor.
            if len(sorted_modalities) > 1 and precomputed_mm_inputs:

                modality_order_dict = {
                    modality: order
                    for order, modality in enumerate(sorted_modalities)
                }

                # Sanity check to make sure each multimodal input has only one
                # modality key.
                for mm_input in precomputed_mm_inputs:
                    assert len(mm_input.modalities) == 1

                # Sort MultiModalKwags to match sorted_mm_positions
                precomputed_mm_inputs = sorted(
                    precomputed_mm_inputs,
                    key=lambda mm_input: modality_order_dict[list(
                        mm_input.modalities)[0]])

            # Apply mm input cache update (and input mapper if necessary).
            sorted_mm_inputs = self.mm_input_mapper_client.process_inputs(
                mm_data=decoder_mm_data,
                mm_hashes=sorted_mm_hashes,
                mm_processor_kwargs=decoder_inputs.mm_processor_kwargs,
                precomputed_mm_inputs=precomputed_mm_inputs,
            )
        else:
            sorted_mm_inputs = None
            sorted_mm_hashes = None
            sorted_mm_positions = None

        return EngineCoreRequest(
            request_id=request_id,
            prompt=decoder_inputs.prompt,
            prompt_token_ids=decoder_inputs.prompt_token_ids,
            mm_inputs=sorted_mm_inputs,
            mm_hashes=sorted_mm_hashes,
            mm_placeholders=sorted_mm_positions,
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
        )

    def _validate_model_inputs(self, inputs: ProcessorInputs):
        if is_encoder_decoder_inputs(inputs):
            # For encoder-decoder multimodal models, the max_prompt_len
            # restricts the decoder prompt length
            prompt_inputs = inputs["decoder" if self.model_config.
                                   is_multimodal_model else "encoder"]
        else:
            prompt_inputs = inputs

        prompt_ids = SingletonInputsAdapter(prompt_inputs).prompt_token_ids

        if prompt_ids is None or len(prompt_ids) == 0:
            raise ValueError("Prompt cannot be empty")

        if self.model_config.is_multimodal_model:
            max_prompt_len = self.model_config.max_model_len

            if len(prompt_ids) > max_prompt_len:
                raise ValueError(
                    f"The prompt (total length {len(prompt_ids)}) is too long "
                    f"to fit into the model (context length {max_prompt_len}). "
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")

            # TODO: Find out how many placeholder tokens are there so we can
            # check that chunked prefill does not truncate them
            # max_batch_len = self.scheduler_config.max_num_batched_tokens
