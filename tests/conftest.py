import contextlib
import gc
import os
import sys
from collections import UserList
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForVision2Seq, AutoTokenizer, BatchEncoding,
                          BatchFeature)

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.config import TokenizerPoolConfig
from vllm.connections import global_http_connection
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         to_enc_dec_tuple_list, zip_enc_dec_prompts)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import SampleLogprobs
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, cuda_device_count_stateless,
                        is_cpu)

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]


def _read_prompts(filename: str) -> List[str]:
    with open(filename, "r") as f:
        prompts = f.readlines()
        return prompts


class _ImageAssetPrompts(TypedDict):
    stop_sign: str
    cherry_blossom: str


if sys.version_info < (3, 9):
    # UserList cannot be subscripted
    class _ImageAssetsBase(UserList):
        pass
else:

    class _ImageAssetsBase(UserList[ImageAsset]):
        pass


class _ImageAssets(_ImageAssetsBase):

    def __init__(self) -> None:
        super().__init__([
            ImageAsset("stop_sign"),
            ImageAsset("cherry_blossom"),
        ])

    def prompts(self, prompts: _ImageAssetPrompts) -> List[str]:
        """
        Convenience method to define the prompt for each test image.

        The order of the returned prompts matches the order of the
        assets when iterating through this object.
        """
        return [prompts["stop_sign"], prompts["cherry_blossom"]]


IMAGE_ASSETS = _ImageAssets()
"""Singleton instance of :class:`_ImageAssets`."""


@pytest.fixture(autouse=True)
def init_test_http_connection():
    # pytest_asyncio may use a different event loop per test
    # so we need to make sure the async client is created anew
    global_http_connection.reuse_client = False


def cleanup():
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not is_cpu():
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


@pytest.fixture
def example_prompts() -> List[str]:
    prompts = []
    for filename in _TEST_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


class DecoderPromptType(Enum):
    """For encoder/decoder models only."""
    CUSTOM = 1
    NONE = 2
    EMPTY_STR = 3


@pytest.fixture
def example_encoder_decoder_prompts(
) -> Dict[DecoderPromptType, List[ExplicitEncoderDecoderPrompt]]:
    '''
    Returns an encoder prompt list and a decoder prompt list, wherein each pair
    of same-index entries in both lists corresponds to an (encoder prompt,
    decoder prompt) tuple.

    Returns:
    
    * Encoder prompt list
    * Decoder prompt list (reverse of encoder prompt list)
    '''

    encoder_prompts = []
    for filename in _TEST_PROMPTS:
        encoder_prompts += _read_prompts(filename)

    custom_decoder_prompts = encoder_prompts[::-1]
    empty_str_decoder_prompts = [""] * len(encoder_prompts)
    none_decoder_prompts = [None] * len(encoder_prompts)

    # NONE decoder prompt type
    return {
        DecoderPromptType.NONE:
        zip_enc_dec_prompts(encoder_prompts, none_decoder_prompts),
        DecoderPromptType.EMPTY_STR:
        zip_enc_dec_prompts(encoder_prompts, empty_str_decoder_prompts),
        DecoderPromptType.CUSTOM:
        zip_enc_dec_prompts(encoder_prompts, custom_decoder_prompts),
    }


@pytest.fixture
def example_long_prompts() -> List[str]:
    prompts = []
    for filename in _LONG_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture(scope="session")
def image_assets() -> _ImageAssets:
    return IMAGE_ASSETS


_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class HfRunner:

    def wrap_device(self, input: _T) -> _T:
        if not is_cpu():
            return input.to("cuda")
        else:
            return input.to("cpu")

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        is_embedding_model: bool = False,
        is_vision_model: bool = False,
        is_encoder_decoder_model: bool = False,
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name

        if is_embedding_model:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer
            self.model = self.wrap_device(
                SentenceTransformer(
                    model_name,
                    device="cpu",
                ).to(dtype=torch_dtype))
        else:
            if is_vision_model:
                auto_cls = AutoModelForVision2Seq
            elif is_encoder_decoder_model:
                auto_cls = AutoModelForSeq2SeqLM
            else:
                auto_cls = AutoModelForCausalLM

            model_kwargs = model_kwargs if model_kwargs is not None else {}
            self.model = self.wrap_device(
                auto_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    **model_kwargs,
                ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        try:
            # don't put this import at the top level
            # it will call torch.cuda.device_count()
            from transformers import AutoProcessor  # noqa: F401
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
        **kwargs: Any,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        if images:
            assert len(prompts) == len(images)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

            inputs = self.processor(**processor_kwargs)

            output_ids = self.model.generate(
                **self.wrap_device(inputs),
                use_cache=True,
                **kwargs,
            )
            output_str = self.processor.batch_decode(
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
        images: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> List[Tuple[List[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                images=images,
                                **kwargs)

        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
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
        images: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> List[List[torch.Tensor]]:
        all_logprobs: List[List[torch.Tensor]] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

            inputs = self.processor(**processor_kwargs)

            output = self.model.generate(
                **self.wrap_device(inputs),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            seq_logprobs: List[torch.Tensor] = []
            for hidden_states in output.hidden_states:
                last_hidden_states = hidden_states[-1][0]
                logits = torch.matmul(
                    last_hidden_states,
                    self.model.get_output_embeddings().weight.t(),
                )
                if self.model.get_output_embeddings().bias is not None:
                    logits += self.model.get_output_embeddings(
                    ).bias.unsqueeze(0)
                logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                seq_logprobs.append(logprobs)
            all_logprobs.append(seq_logprobs)
        return all_logprobs

    def _hidden_states_to_logprobs(
        self,
        hidden_states,
        num_logprobs,
    ) -> Tuple[List[Dict[int, float]], int]:
        seq_logprobs: List[torch.Tensor] = []
        output_len = len(hidden_states)
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states,
                self.model.get_output_embeddings().weight.t(),
            )
            if getattr(self.model.get_output_embeddings(), "bias",
                       None) is not None:
                logits += self.model.get_output_embeddings().bias.unsqueeze(0)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            seq_logprobs.append(logprobs)

        # convert to dict
        seq_logprobs_lst: List[Dict[int, float]] = []
        for tok_idx, tok_logprobs in enumerate(seq_logprobs):
            # drop prompt logprobs
            if tok_idx == 0:
                tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
            topk = tok_logprobs.topk(num_logprobs)

            tok_logprobs_dct = {}
            for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                tok_logprobs_dct[token_id.item()] = logprob.item()

            seq_logprobs_lst.append(tok_logprobs_dct)

        return (
            seq_logprobs_lst,
            output_len,
        )

    def generate_greedy_logprobs_limit(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
        images: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> List[Tuple[List[int], str, List[Dict[int, float]]]]:
        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

            inputs = self.processor(**processor_kwargs)

            output = self.model.generate(
                **self.wrap_device(inputs),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = len(seq_logprobs_lst)
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_encoder_decoder_greedy_logprobs_limit(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        **kwargs: Any,
    ) -> List[Tuple[List[int], str, List[Dict[int, float]]]]:
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for (encoder_prompt,
             decoder_prompt) in to_enc_dec_tuple_list(encoder_decoder_prompts):
            encoder_input_ids = self.wrap_device(
                self.tokenizer(encoder_prompt, return_tensors="pt").input_ids)
            decoder_input_ids = (
                None if decoder_prompt is None else self.wrap_device(
                    self.tokenizer(decoder_prompt,
                                   return_tensors="pt").input_ids))

            output = self.model.generate(
                encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.decoder_hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        return self.model.encode(prompts)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


class VllmRunner:

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len: int = 1024,
        dtype: str = "half",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        **kwargs,
    ) -> None:
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name,
            trust_remote_code=True,
            dtype=dtype,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
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
        images: Optional[List[Image.Image]] = None,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        if images is not None:
            assert len(prompts) == len(images)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["multi_modal_data"] = {"image": image}

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)

        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: List[List[int]] = []
            req_sample_output_strs: List[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    def _final_steps_generate_w_logprobs(
        self,
        req_outputs: List[RequestOutput],
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        outputs: List[Tuple[List[int], str, Optional[SampleLogprobs]]] = []
        for req_output in req_outputs:
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = sample.token_ids
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        images: Optional[List[Image.Image]] = None,
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        assert sampling_params.logprobs is not None

        if images is not None:
            assert len(prompts) == len(images)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["multi_modal_data"] = {"image": image}

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params)
        return self._final_steps_generate_w_logprobs(req_outputs)

    def generate_encoder_decoder_w_logprobs(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        sampling_params: SamplingParams,
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        '''
        Logprobs generation for vLLM encoder/decoder models
        '''

        assert sampling_params.logprobs is not None
        req_outputs = self.model.generate(encoder_decoder_prompts,
                                          sampling_params=sampling_params)
        return self._final_steps_generate_w_logprobs(req_outputs)

    def generate_greedy(
        self,
        prompts: List[str],
        max_tokens: int,
        images: Optional[List[Image.Image]] = None,
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
        images: Optional[Union[List[Image.Image],
                               List[List[Image.Image]]]] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        greedy_logprobs_params = SamplingParams(temperature=0.0,
                                                max_tokens=max_tokens,
                                                logprobs=num_logprobs,
                                                stop_token_ids=stop_token_ids)
        outputs = self.generate_w_logprobs(prompts,
                                           greedy_logprobs_params,
                                           images=images)

        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_encoder_decoder_greedy_logprobs(
        self,
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        greedy_logprobs_params = SamplingParams(temperature=0.0,
                                                use_beam_search=False,
                                                max_tokens=max_tokens,
                                                logprobs=num_logprobs)
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

        outputs = self.generate_encoder_decoder_w_logprobs(
            encoder_decoder_prompts, greedy_logprobs_params)

        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_beam_search(
        self,
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


def get_tokenizer_pool_config(tokenizer_group_type):
    if tokenizer_group_type is None:
        return None
    if tokenizer_group_type == "ray":
        return TokenizerPoolConfig(pool_size=1,
                                   pool_type="ray",
                                   extra_config={})
    if isinstance(tokenizer_group_type, type):
        return TokenizerPoolConfig(pool_size=1,
                                   pool_type=tokenizer_group_type,
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
def num_gpus_available():
    """Get number of GPUs without initializing the CUDA context
    in current process."""

    return cuda_device_count_stateless()
