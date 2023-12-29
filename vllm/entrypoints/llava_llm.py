from vllm.engine.llava_engine import LLaVAEngine
from PIL import Image
import requests
import base64
from io import BytesIO
import numpy as np
from typing import List, Optional, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


class LLaVA:

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            **kwargs,
        )
        self.llm_engine = LLaVAEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

        self.image_token_index = self.llm_engine.model_config.hf_config.image_token_index
        self.image_token = self.get_tokenizer().decode(self.image_token_index)

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def get_image_processor(self):
        return self.llm_engine.image_processor

    def set_image_processor(self, image_processor):
        self.llm_engine.image_processor = image_processor

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        if (prompts is not None and prompt_token_ids is not None
                and len(prompts) != len(prompt_token_ids)):
            raise ValueError("The lengths of prompts and prompt_token_ids "
                             "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # process images
        if images is None:
            images = []
        elif not isinstance(images, list):
            images = [images]
        _images = []
        image_id = 0
        for image in images:
            if isinstance(image, str):
                if image.startswith("http"):
                    _images.append(
                        Image.open(requests.get(image, stream=True).raw))
                elif image.startswith("data:"):
                    _images.append(
                        Image.open(
                            BytesIO(base64.b64decode(image.split(",")[1]))))
                elif image.startswith("/"):
                    _images.append(Image.open(image))
                else:
                    _images.append(Image.open(BytesIO(
                        base64.b64decode(image))))
            elif isinstance(image, Image.Image):
                _images.append(image)
            else:
                raise ValueError("image must be str or PIL.Image")

        # Add requests to the engine.
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]

            image_token_num = 0
            if prompt is not None:
                image_token_num = prompt.count(self.image_token)
            if token_ids is not None:
                _image_token_num = np.sum(
                    np.asarray(token_ids) == self.image_token_index)
                if image_token_num != _image_token_num:
                    raise ValueError("image_token_num != _image_token_num")
                else:
                    image_token_num = _image_token_num
            if image_token_num > 0:
                assert image_id + image_token_num <= len(
                    _images
                ), " The input provided to the model are wrong. The number of image tokens is not equal to the number of images provided."
                images = _images[image_id:image_id + image_token_num]
                image_id += image_token_num
            else:
                images = None

            self._add_request(prompt, sampling_params, token_ids, images)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
        images: Optional[List[Image.Image]] = None,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids,
                                    images=images)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
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
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return outputs
