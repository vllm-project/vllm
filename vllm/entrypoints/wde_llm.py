from typing import List, Optional, Sequence, Union, cast

from tqdm import tqdm

from vllm.engine.wde_engine import LLMEngine
from vllm.inputs.prefill_only.data import Params
from vllm.inputs.prefill_only.data import TextOnlyInputs as PromptInputs
from vllm.inputs.prefill_only.data import ValidationError
from vllm.logger import init_logger
from vllm.model_executor.prefill_only.engine_io import RequestOutput
from vllm.utils import Counter

logger = init_logger(__name__)


class LLM:

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size",
                               "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore.")
        engine_args = dict(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def encode(
        self,
        inputs: Union[Union[PromptInputs, Sequence[PromptInputs]],
                      Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[Params, Sequence[Params]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        inputs = cast(Union[PromptInputs, Sequence[PromptInputs]], inputs)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = Params()

        self._validate_and_add_requests(
            inputs=inputs,
            params=pooling_params,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return outputs

    def _validate_and_add_requests(
        self,
        inputs: Union[PromptInputs, Sequence[PromptInputs]],
        params: Optional[Union[Params, Sequence[Params]]] = None,
    ) -> None:

        params = params if params is not None else Params()
        if isinstance(inputs, PromptInputs):
            inputs = [inputs]

        # Add requests to the engine.
        for i, request_inputs in enumerate(inputs):
            try:
                self._add_request(
                    request_inputs,
                    params[i] if isinstance(params, Sequence) else params)
            except ValidationError as e:
                raise e

    def _add_request(
        self,
        inputs: PromptInputs,
        params: Params,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, inputs, params)

    def _run_engine(self, *, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
