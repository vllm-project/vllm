from typing import List, Optional

from tqdm import tqdm

from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import ServerArgs
from cacheflow.server.llm_server import LLMServer
from cacheflow.utils import Counter


class LLM:

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
