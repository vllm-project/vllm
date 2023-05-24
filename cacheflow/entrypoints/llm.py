from typing import List, Optional

from tqdm import tqdm

from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import ServerArgs
from cacheflow.server.llm_server import LLMServer
from cacheflow.utils import Counter


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, an LLM model (possibly distributed across
    multiple GPUs), and GPU memory space allocated for intermediate states (aka
    KV cache). Given a batch of prompts and sampling parameters, this class
    generates texts from the LLM model, using an intelligent batching mechanism
    and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `LLMServer` class instead.

    Args:
        model: The name or path of a huggingface transformers model.
        tensor_parallel_size: The number of GPUs to use for distributed execution.
        dtype: The data type to use for the model weights and activations.
            Currently, we support `float16`, and `bfloat16`. If `default`, we
            refer to torch_dtype in the model config and use `float16` for
            `float16` and `float32` models, and `bfloat16` for `bfloat16` models.
        seed: The seed to initialize the random states.

    For more arguments, see `ServerArgs`.
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "default",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        server_args = ServerArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.llm_server = LLMServer.from_server_args(server_args)
        self.request_counter = Counter()

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()
        # Initialize tqdm.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Processed prompts")

        # Add requests to the server.
        for i in range(len(prompts)):
            prompt = prompts[i]
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            request_id = str(next(self.request_counter))
            self.llm_server.add_request(request_id, prompt, sampling_params,
                                        token_ids)

        # Run the server.
        outputs: List[RequestOutput] = []
        while self.llm_server.has_unfinished_requests():
            step_outputs = self.llm_server.step()
            for output in step_outputs:
                if output.done:
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        return outputs
