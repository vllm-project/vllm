import os
from typing import List, Optional, Tuple

import pytest
import torch
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]


def _read_prompts(filename: str) -> List[str]:
    with open(filename, "r") as f:
        prompts = f.readlines()
        return prompts


@pytest.fixture
def example_prompts() -> List[str]:
    prompts = []
    for filename in _TEST_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture
def example_long_prompts() -> List[str]:
    prompts = []
    for filename in _LONG_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


class HfRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
    ) -> None:
        assert dtype in _STR_DTYPE_TO_TORCH_DTYPE
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).cuda()
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer = get_tokenizer(tokenizer_name, trust_remote_code=True)

    def generate(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Tuple[List[int], str]]:
        outputs: List[Tuple[List[int], str]] = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output_ids = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                **kwargs,
            )
            output_str = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_ids = output_ids.cpu().tolist()
            outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            outputs[i] = (output_ids[0], output_str[0])
        return outputs

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
                                num_return_sequences=beam_width)
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            for j in range(len(output_ids)):
                output_ids[j] = [
                    x for x in output_ids[j]
                    if x != self.tokenizer.pad_token_id
                ]
            outputs[i] = (output_ids, output_str)
        return outputs

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[List[torch.Tensor]]:
        all_logprobs = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                input_ids.cuda(),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            seq_logprobs = []
            for hidden_states in output.hidden_states:
                last_hidden_states = hidden_states[-1][0]
                logits = torch.matmul(
                    last_hidden_states,
                    self.model.get_output_embeddings().weight.t(),
                )
                if self.model.get_output_embeddings().bias is not None:
                    logits += self.model.get_output_embeddings(
                    ).bias.unsqueeze(0)
                logprobs = torch.nn.functional.log_softmax(logits,
                                                           dim=-1,
                                                           dtype=torch.float32)
                seq_logprobs.append(logprobs)
            all_logprobs.append(seq_logprobs)
        return all_logprobs


@pytest.fixture
def hf_runner():
    return HfRunner


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=0,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[Tuple[List[int], str]]:
        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params)
        outputs = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids = []
            req_sample_output_strs = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[Tuple[List[int], str]]:
        assert sampling_params.logprobs is not None

        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params)
        outputs = []
        for req_output in req_outputs:
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs))
        return outputs

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts, greedy_params)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
    ) -> List[Tuple[List[int], str]]:
        greedy_logprobs_params = SamplingParams(temperature=0.0,
                                                max_tokens=max_tokens,
                                                logprobs=num_logprobs)
        outputs = self.generate_w_logprobs(prompts, greedy_logprobs_params)

        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[int], str]]:
        beam_search_params = SamplingParams(n=beam_width,
                                            use_beam_search=True,
                                            temperature=0.0,
                                            max_tokens=max_tokens)
        outputs = self.generate(prompts, beam_search_params)
        return outputs


@pytest.fixture
def vllm_runner():
    return VllmRunner
