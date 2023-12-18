
import gc
from typing import List, Optional, Tuple

import pytest
import ray
import torch
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm.transformers_utils.tokenizer import get_tokenizer

_TEST_PROMPTS = [
    # pylint: disable=line-too-long
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]


def cleanup():
    # Revert to torch default after vllm modifications
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()


@pytest.fixture
def example_prompts() -> List[str]:
    return _TEST_PROMPTS


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
            device_map={"": 0},
        )
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
    yield HfRunner
    cleanup()


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        dtype: str = "half",
        enable_cuda_graph: bool = False,
        cuda_graph_max_context_len: int = 5000,
        cuda_graph_cache_size: int = 10,
        tensor_parallel_size: int = 1,
        flash_style: bool = False,
        max_chunked_prefill_len: int = -1,
        max_num_prompt_seqs: int = 1000,
        max_num_batched_tokens: int = 4096,
        worker_use_ray: bool = False,
        input_padding_size: int = 8,
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=0,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_max_context_len=cuda_graph_max_context_len,
            cuda_graph_cache_size=cuda_graph_cache_size,
            tensor_parallel_size=tensor_parallel_size,
            flash_style=flash_style,
            block_size=32,
            max_chunked_prefill_len=max_chunked_prefill_len,
            max_num_prompt_seqs=max_num_prompt_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            worker_use_ray=worker_use_ray,
            input_padding_size=input_padding_size)

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        return_output_only: bool = False,
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
                if return_output_only:
                    req_sample_output_ids.append(output_ids)
                    req_sample_output_strs.append(output_str)
                else:
                    req_sample_output_ids.append(prompt_ids + output_ids)
                    req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
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


@pytest.fixture
def vllm_runner():
    yield VllmRunner
    cleanup()


@pytest.fixture
def setup_cuda_graph(model_name="facebook/opt-125m", cache_size=8):
    vllm_model = VllmRunner(model_name,
                            dtype="half",
                            enable_cuda_graph=True,
                            cuda_graph_max_context_len=64,
                            cuda_graph_cache_size=cache_size)
    nn_model = vllm_model.model.llm_engine.workers[0].model
    cuda_graph = vllm_model.model.llm_engine.workers[0].captured_model
    worker = vllm_model.model.llm_engine.workers[0]
    yield vllm_model, nn_model, cuda_graph, worker
    del vllm_model
    del nn_model
    del cuda_graph
    del worker
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
