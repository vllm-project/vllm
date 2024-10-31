import time
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from vllm.config import ModelConfig, LoRAConfig, SchedulerConfig, ParallelConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs,
                         EncoderDecoderLLMInputs, InputRegistry, PromptType)
from vllm.inputs.preprocess import InputPreprocessor
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

from vllm.v1.engine import (DetokenizerRequest,
                            EngineCoreRequest)

class Processor:
    def __init__(
        self, 
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        lora_config: Optional[LoRAConfig],
        input_registry: InputRegistry = INPUT_REGISTRY,
    ):

        self.model_config = model_config
        self.lora_config = lora_config

        assert not model_config.skip_tokenizer_init

        self.tokenizer = init_tokenizer_from_configs(
            model_config=model_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            enable_lora=bool(lora_config))
        
        # Ping the tokenizer to ensure liveness if it runs in a
        # different process.
        self.tokenizer.ping()

        self.generation_config_fields = _load_generation_config_dict(
            model_config)
        self.input_preprocessor = InputPreprocessor(model_config,
                                                    self.tokenizer)
        self.input_processor = input_registry.create_input_processor(
            model_config)
    
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

        assert isinstance(params, SamplingParams)
        sampling_params = params.clone()
        sampling_params.update_from_generation_config(
            self.generation_config_fields, eos_token_id)

        # Make Request for Detokenizer.
        detokenizer_request = DetokenizerRequest(
            request_id, processed_inputs.get("prompt"),
            processed_inputs.get("prompt_token_ids"),
            sampling_params.skip_special_tokens,
            sampling_params.spaces_between_special_tokens,
            sampling_params.output_kind)

        # Make Request for EngineCore.
        engine_core_request = EngineCoreRequest(
            request_id, processed_inputs.get("prompt"),
            processed_inputs.get("prompt_token_ids"), sampling_params,
            eos_token_id, arrival_time, lora_request)

        return detokenizer_request, engine_core_request
    
    def _validate_model_inputs(self, inputs: Union[DecoderOnlyInputs,
                                                   EncoderDecoderLLMInputs]):
        prompt_ids = inputs.get("prompt_token_ids")
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

def _load_generation_config_dict(model_config: ModelConfig) -> Dict[str, Any]:
    config = try_get_generation_config(
        model_config.model,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )

    if config is None:
        return {}

    return config.to_diff_dict()