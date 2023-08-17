#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional, Union, List

from vllm import LLM, SamplingParams, RequestOutput, EngineArgs
from tqdm import tqdm

from vllm.engine.mllm_engine import MLLMEngine, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vllm.utils import Counter


class MLLM(LLM):
    def __init__(
            self,
            model: str,
            tokenizer: Optional[str] = None,
            tokenizer_mode: str = "auto",
            trust_remote_code: bool = False,
            tensor_parallel_size: int = 1,
            dtype: str = "auto",
            seed: int = 0,
            **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        CLIP_MODEL_MAP={}
        CLIP_MODEL_MAP.update({"openai/clip-vit-large-patch14":f"{os.path.abspath(model)}/clip-vit-large-patch14"})

        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.mllm_engine = MLLMEngine.from_engine_args(engine_args)
        self.mllm_engine.initialize_vision_tokenizer()
        self.request_counter = Counter()

    def generate(
            self,
            prompts: Optional[Union[str, List[str]]] = None,
            images: Optional[Union[dict, List[dict]]] = None,
            sampling_params: Optional[SamplingParams] = None,
            prompt_token_ids: Optional[List[List[int]]] = None,
            use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_tokes_t = time.time()n_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        assert len(prompts) == len(
            images), f"The number of images entered should be the same as the number of text，get image number is " \
                     f"{len(images)} but text number is {len(prompts)}." \
                     "if image is None, please use {} placeholder。"

        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)

        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            image = images[i]
            self._add_request(prompt, sampling_params, token_ids, image)

        result = self._run_engine(use_tqdm)
        return result

    def _prompt_image(self, prompts: List[str], images: List[dict], is_only_prompts=False) -> Union[List[str], None]:
        assert len(prompts) == len(
            images), f"The number of images entered should be the same as the number of text，get image number is " \
                     f"{len(images)} but text number is {len(prompts)}." \
                     "if image is None, please use {} placeholder。"
        if is_only_prompts:
            results = []
            for prompt, image in zip(prompts, images):
                if image:
                    img_data = image.get("image_src")
                    img_type = image.get("src_type")
                    img_prompt = " ".join(
                        [DEFAULT_IM_START_TOKEN, img_type, DEFAULT_IMAGE_PATCH_TOKEN, img_data, DEFAULT_IM_END_TOKEN])
                    prompt += img_prompt
                results.append(prompt)
            return results

    def _add_request(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            prompt_token_ids: Optional[List[int]],
            image: Optional[dict] = None
    ) -> None:
        request_id = str(next(self.request_counter))

        self.mllm_engine.add_request(request_id, prompt, image, sampling_params,
                                     prompt_token_ids)


    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.mllm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        while self.mllm_engine.has_unfinished_requests():
            step_outputs = self.mllm_engine.step()
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
