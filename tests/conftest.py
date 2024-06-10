import contextlib
import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          LlavaConfig, LlavaForConditionalGeneration)

from tests.nm_utils.logging import make_logger
from vllm import LLM, SamplingParams
from vllm.config import TokenizerPoolConfig, VisionLanguageConfig
from vllm.distributed import destroy_model_parallel
from vllm.inputs import PromptInputs
from vllm.logger import init_logger
from vllm.sequence import MultiModalData

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]

# Multi modal related
_PIXEL_VALUES_FILES = [
    os.path.join(_TEST_DIR, "images", filename) for filename in
    ["stop_sign_pixel_values.pt", "cherry_blossom_pixel_values.pt"]
]
_IMAGE_FEATURES_FILES = [
    os.path.join(_TEST_DIR, "images", filename) for filename in
    ["stop_sign_image_features.pt", "cherry_blossom_image_features.pt"]
]
_IMAGE_FILES = [
    os.path.join(_TEST_DIR, "images", filename)
    for filename in ["stop_sign.jpg", "cherry_blossom.jpg"]
]
_IMAGE_PROMPTS = [
    "<image>\nUSER: What's the content of the image?\nASSISTANT:",
    "<image>\nUSER: What is the season?\nASSISTANT:"
]
assert len(_PIXEL_VALUES_FILES) == len(_IMAGE_FEATURES_FILES) == len(
    _IMAGE_FILES) == len(_IMAGE_PROMPTS)


def _read_prompts(filename: str) -> List[str]:
    with open(filename, "r") as f:
        prompts = f.readlines()
        return prompts


def cleanup():
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    if request.node.get_closest_marker("skip_global_cleanup"):
        return False

    return True


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup()


@pytest.fixture(scope="session")
def hf_image_prompts() -> List[str]:
    return _IMAGE_PROMPTS


@pytest.fixture(scope="session")
def hf_images() -> List[Image.Image]:
    return [Image.open(filename) for filename in _IMAGE_FILES]


@pytest.fixture()
def vllm_images(request) -> "torch.Tensor":
    vision_language_config = request.getfixturevalue("model_and_config")[1]
    all_images = []
    if vision_language_config.image_input_type == (
            VisionLanguageConfig.ImageInputType.IMAGE_FEATURES):
        filenames = _IMAGE_FEATURES_FILES
    else:
        filenames = _PIXEL_VALUES_FILES
    for filename in filenames:
        all_images.append(torch.load(filename))
    return torch.concat(all_images, dim=0)


@pytest.fixture()
def vllm_image_prompts(request) -> List[str]:
    vision_language_config = request.getfixturevalue("model_and_config")[1]
    return [
        "<image>" * (vision_language_config.image_feature_size - 1) + p
        for p in _IMAGE_PROMPTS
    ]


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

AutoModelForCausalLM.register(LlavaConfig, LlavaForConditionalGeneration)

_EMBEDDING_MODELS = [
    "intfloat/e5-mistral-7b-instruct",
]


class HfRunner:

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        access_token: Optional[str] = None,
    ) -> None:
        assert dtype in _STR_DTYPE_TO_TORCH_DTYPE
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name

        if model_name in _EMBEDDING_MODELS:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                model_name,
                device="cpu",
            ).to(dtype=torch_dtype).cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                token=access_token,
            ).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            logger.warning(
                "Unable to auto-load processor from HuggingFace for "
                "model %s. Using tokenizer instead.", model_name)
            self.processor = self.tokenizer

    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> List[Tuple[List[int], str]]:
        outputs: List[Tuple[List[int], str]] = []
        if images:
            assert len(prompts) == len(images)
        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

            inputs = self.processor(**processor_kwargs)
            inputs = {
                key: value.cuda() if value is not None else None
                for key, value in inputs.items()
            }

            output_ids = self.model.generate(
                **inputs,
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
        images: Optional["torch.Tensor"] = None,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                images=images)
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

    def generate_greedy_logprobs_limit(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
    ) -> List[Tuple[List[int], str]]:
        all_logprobs = []
        all_output_ids = []
        all_output_strs = []

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
            for _, hidden_states in enumerate(output.hidden_states):
                last_hidden_states = hidden_states[-1][0]
                logits = torch.matmul(
                    last_hidden_states,
                    self.model.get_output_embeddings().weight.t(),
                )
                if getattr(self.model.get_output_embeddings(), "bias",
                           None) is not None:
                    logits += self.model.get_output_embeddings(
                    ).bias.unsqueeze(0)
                logprobs = torch.nn.functional.log_softmax(logits,
                                                           dim=-1,
                                                           dtype=torch.float32)
                seq_logprobs.append(logprobs)

            # convert to dict
            seq_logprobs_lst = []
            for tok_idx, tok_logprobs in enumerate(seq_logprobs):
                # drop prompt logprobs
                if tok_idx == 0:
                    tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
                topk = tok_logprobs.topk(num_logprobs)

                tok_logprobs_dct = {}
                for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                    tok_logprobs_dct[token_id.item()] = logprob.item()

                seq_logprobs_lst.append(tok_logprobs_dct)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = seq_ids.shape[0] - input_ids.shape[1]
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        return self.model.encode(prompts)

    def __del__(self):
        del self.model
        cleanup()


@pytest.fixture
def hf_runner():
    return HfRunner


# UPSTREAM SYNC: needed for nm-automation
class HfRunnerNM(HfRunner):

    def _logprobs_from_generated_hidden_states(self, output):
        """
        generates a list of logprobs from the output of self.model.generate()
        """
        seq_logprobs = []
        for _, hidden_states in enumerate(output.hidden_states):
            last_hidden_states = hidden_states[-1][0]
            logits = torch.matmul(
                last_hidden_states,
                self.model.get_output_embeddings().weight.t(),
            )
            if self.model.get_output_embeddings().bias is not None:
                logits += self.model.get_output_embeddings().bias.unsqueeze(0)
            logprobs = torch.nn.functional.log_softmax(logits,
                                                       dim=-1,
                                                       dtype=torch.float32)
            seq_logprobs.append(logprobs)
        return seq_logprobs

    def generate_greedy_logprobs_nm(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
    ) -> List[Tuple[List[int], str, List[Dict]]]:
        all_logprobs = []
        all_output_ids = []
        all_output_strs = []

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

            seq_logprobs = self._logprobs_from_generated_hidden_states(output)

            # convert to dict
            seq_logprobs_lst = []
            for tok_idx, tok_logprobs in enumerate(seq_logprobs):
                # drop prompt logprobs
                if tok_idx == 0:
                    tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
                topk = tok_logprobs.topk(num_logprobs)

                tok_logprobs_dct = {}
                for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                    tok_logprobs_dct[token_id.item()] = logprob.item()

                seq_logprobs_lst.append(tok_logprobs_dct)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = seq_ids.shape[0] - input_ids.shape[1]
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "�", "</s>"]

    def _decode_token_by_position_index(
            self,
            token_index: int,
            token_ids: List[int],
            clean_up_tokenization_spaces: bool = True,
            ignore_special_tokens: bool = False) -> str:
        """
        helper function to calculate a token string at the specified index
        based on the previous tokens.

        :param token_index: position in the list of token ids where you would
            like the token string
        :param token_ids: the list of token ids
        :param clean_up_tokenization_spaces: option to pass to
            `tokenizer.decode()`
        :param ignore_special_tokens: converts hard coded special tokens to
            an empty string
        """
        lookback = 4
        prior_str = self.tokenizer.decode(
            token_ids[token_index - lookback:token_index - 1],
            clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        current_str = self.tokenizer.decode(
            token_ids[token_index - lookback:token_index],
            clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        token = current_str[-(len(current_str) - len(prior_str)):]
        if ignore_special_tokens and token in self.SPECIAL_TOKENS:
            token = ""
        return token

    def generate_greedy_logprobs_nm_use_tokens(
        self,
        prompts: List[str],
        max_tokens: int,
        topk_logprobs_count: int,
        ignore_special_tokens: bool = False
    ) -> List[Tuple[List[int], str, List[Dict]]]:
        all_logprobs = []
        all_output_tokens = []
        all_output_strs = []

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

            seq_logprobs = self._logprobs_from_generated_hidden_states(output)

            # convert sequence of logprobs to a dict keyed on the selected token
            seq_ids = output.sequences[0]
            input_len = input_ids.shape[1]
            seq_logprobs_lst = list()
            output_tokens = list()

            for tok_idx, tok_logprobs in enumerate(seq_logprobs):
                # drop prompt logprobs
                if tok_idx == 0:
                    tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
                topk = tok_logprobs.topk(topk_logprobs_count)

                tok_logprobs_dct = {}
                # add 1 to the index here to be in sync with the skipped prompt
                # in the logprobs
                indexed_seq = seq_ids[:input_len + tok_idx + 1].tolist()
                token_str = self._decode_token_by_position_index(
                    input_len + tok_idx + 1,
                    indexed_seq,
                    clean_up_tokenization_spaces=False,
                    ignore_special_tokens=True)
                output_tokens.append(token_str)
                for alt_token_id, logprob in zip(topk.indices[0],
                                                 topk.values[0]):
                    # replace the token_str at the tok_idx with alt_token_id.
                    indexed_seq[-1] = alt_token_id
                    # then run decode again
                    logprob_str = self._decode_token_by_position_index(
                        input_len + tok_idx + 1,
                        indexed_seq,
                        clean_up_tokenization_spaces=False,
                        ignore_special_tokens=True)
                    tok_logprobs_dct[logprob_str] = logprob.item()

                seq_logprobs_lst.append(tok_logprobs_dct)

            all_logprobs.append(seq_logprobs_lst)

            all_output_tokens.append(output_tokens)
            all_output_strs.append("".join(output_tokens))

        outputs = zip(all_output_tokens, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def __del__(self):
        del self.model
        cleanup()


@pytest.fixture
def hf_runner_nm():
    return HfRunnerNM


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len=1024,
        dtype: str = "half",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space=4,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            swap_space=swap_space,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
            **kwargs,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional["torch.Tensor"] = None,
    ) -> List[Tuple[List[int], str]]:
        if images is not None:
            assert len(prompts) == images.shape[0]

        prompt_inputs: List[PromptInputs] = []
        for i, prompt in enumerate(prompts):
            image = None if images is None else images[i:i + 1]
            mm_data = None if image is None else MultiModalData(
                type=MultiModalData.Type.IMAGE,
                data=image,
            )

            prompt_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
            })

        req_outputs = self.model.generate(prompt_inputs,
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
        images: Optional[torch.Tensor] = None,
    ) -> List[Tuple[List[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts, greedy_params, images=images)
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

    def encode(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.encode(prompts)
        outputs = []
        for req_output in req_outputs:
            embedding = req_output.outputs.embedding
            outputs.append(embedding)
        return outputs

    def __del__(self):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


# UPSTREAM SYNC: needed for nm-automation
class VllmRunnerNm(VllmRunner):

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

    def __del__(self):
        del self.model
        cleanup()


@pytest.fixture
def vllm_runner_nm():
    return VllmRunnerNm


def get_tokenizer_pool_config(tokenizer_group_type):
    if tokenizer_group_type is None:
        return None
    if tokenizer_group_type == "ray":
        return TokenizerPoolConfig(pool_size=1,
                                   pool_type="ray",
                                   extra_config={})
    raise ValueError(f"Unknown tokenizer_group_type: {tokenizer_group_type}")


@pytest.fixture()
def temporary_enable_log_propagate():
    import logging
    logger = logging.getLogger("vllm")
    logger.propagate = True
    yield
    logger.propagate = False


@pytest.fixture()
def caplog_vllm(temporary_enable_log_propagate, caplog):
    # To capture vllm log, we should enable propagate=True temporarily
    # because caplog depends on logs propagated to the root logger.
    yield caplog


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    return make_logger("vllm_test")
