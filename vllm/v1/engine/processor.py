import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from vllm.config import LoRAConfig, ModelConfig
from vllm.inputs import (INPUT_REGISTRY, InputRegistry, ProcessorInputs,
                         PromptType, SingletonInputsAdapter)
from vllm.inputs.parse import is_encoder_decoder_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup
from vllm.v1.engine import DetokenizerRequest, EngineCoreRequest


class Processor:

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: Optional[LoRAConfig],
        tokenizer: BaseTokenizerGroup,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        self.model_config = model_config
        self.lora_config = lora_config
        self.tokenizer = tokenizer

        self.generation_config_fields = _load_generation_config_dict(
            model_config)
        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer,
                                                    mm_registry)
        self.input_processor = input_registry.create_input_processor(
            model_config)

    # TODO: run in an ThreadpoolExecutor or BackgroundProcess.
    # This ideally should releases the GIL, so we should not block the
    # asyncio loop while this is running.
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> Tuple[DetokenizerRequest, EngineCoreRequest]:

        # TODO(woosuk): Support embedding mode.
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

        # Make Request for Detokenizer.
        detokenizer_request = DetokenizerRequest(
            request_id,
            decoder_inputs.prompt,
            decoder_inputs.prompt_token_ids,
            sampling_params.skip_special_tokens,
            sampling_params.spaces_between_special_tokens,
            sampling_params.output_kind,
            sampling_params.stop,
            sampling_params.include_stop_str_in_output,
        )

        # Make Request for EngineCore.
        engine_core_request = EngineCoreRequest(
            request_id,
            decoder_inputs.prompt,
            decoder_inputs.prompt_token_ids,
            decoder_inputs.multi_modal_data,
            decoder_inputs.multi_modal_placeholders,
            decoder_inputs.mm_processor_kwargs,
            sampling_params,
            eos_token_id,
            arrival_time,
            lora_request,
        )

        return detokenizer_request, engine_core_request

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


def _load_generation_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    config = try_get_generation_config(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if config is None:
        return {}

    return config.to_diff_dict()
