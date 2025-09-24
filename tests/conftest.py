# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa

from tblib import pickling_support

# Install support for pickling exceptions so that we can nicely propagate
# failures from tests running in a subprocess.
# This should be run before any custom exception subclasses are defined.
pickling_support.install()

import http.server
import json
import math
import mimetypes
import os
import socket
import tempfile
import threading
from collections.abc import Generator
from contextlib import nullcontext
from enum import Enum
from typing import Any, Callable, Optional, TypedDict, TypeVar, Union, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from tests.models.utils import (TokensTextLogprobs,
                                TokensTextLogprobsPromptLogprobs)
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config.model import (ConvertOption, RunnerOption,
                               _get_and_verify_dtype)
from vllm.connections import global_http_connection
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.multimodal.utils import fetch_image
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils import set_default_torch_num_threads

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]
_SYS_MSG = os.path.join(_TEST_DIR, "system_messages", "sonnet3.5_nov2024.txt")

_M = TypeVar("_M")

_PromptMultiModalInput = Union[list[_M], list[list[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]


def _read_prompts(filename: str) -> list[str]:
    with open(filename) as f:
        prompts = f.readlines()
        return prompts


class ImageAssetPrompts(TypedDict):
    stop_sign: str
    cherry_blossom: str


class ImageTestAssets(list[ImageAsset]):

    def __init__(self) -> None:
        super().__init__([
            ImageAsset("stop_sign"),
            ImageAsset("cherry_blossom"),
        ])

    def prompts(self, prompts: ImageAssetPrompts) -> list[str]:
        """
        Convenience method to define the prompt for each test image.

        The order of the returned prompts matches the order of the
        assets when iterating through this object.
        """
        return [prompts["stop_sign"], prompts["cherry_blossom"]]


class VideoAssetPrompts(TypedDict):
    baby_reading: str


class VideoTestAssets(list[VideoAsset]):

    def __init__(self) -> None:
        super().__init__([
            VideoAsset("baby_reading"),
        ])

    def prompts(self, prompts: VideoAssetPrompts) -> list[str]:
        return [prompts["baby_reading"]]


class AudioAssetPrompts(TypedDict):
    mary_had_lamb: str
    winning_call: str


class AudioTestAssets(list[AudioAsset]):

    def __init__(self) -> None:
        super().__init__([
            AudioAsset("mary_had_lamb"),
            AudioAsset("winning_call"),
        ])

    def prompts(self, prompts: AudioAssetPrompts) -> list[str]:
        return [prompts["mary_had_lamb"], prompts["winning_call"]]


IMAGE_ASSETS = ImageTestAssets()
"""Singleton instance of {class}`ImageTestAssets`."""
VIDEO_ASSETS = VideoTestAssets()
"""Singleton instance of {class}`VideoTestAssets`."""
AUDIO_ASSETS = AudioTestAssets()
"""Singleton instance of {class}`AudioTestAssets`."""


@pytest.fixture(scope="function", autouse=True)
def cleanup_VLLM_USE_V1(monkeypatch):
    """
    The V1 oracle sets "VLLM_USE_V1" during loading. This means
    that each invocation of a test change the env variable.

    If we touch "VLLM_USE_V1" with monkeypatch, then any changes
    made during the test run by vLLM will be cleaned up.

    This fixture is used by every test.
    """

    # If VLLM_USE_V1 is not set, set then delete. This will
    # cause monkeypatch to clean up VLLM_USE_V1 upon exit
    # if VLLM modifies the value of envs.VLLM_USE_V1.
    if "VLLM_USE_V1" not in os.environ:
        monkeypatch.setenv("VLLM_USE_V1", "")
        monkeypatch.delenv("VLLM_USE_V1")


@pytest.fixture(autouse=True)
def init_test_http_connection():
    # pytest_asyncio may use a different event loop per test
    # so we need to make sure the async client is created anew
    global_http_connection.reuse_client = False


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield
    torch._dynamo.reset()


@pytest.fixture
def example_prompts() -> list[str]:
    prompts = []
    for filename in _TEST_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture
def example_system_message() -> str:
    with open(_SYS_MSG) as f:
        return f.read()


class DecoderPromptType(Enum):
    """For encoder/decoder models only."""
    CUSTOM = 1
    NONE = 2
    EMPTY_STR = 3


@pytest.fixture
def example_long_prompts() -> list[str]:
    prompts = []
    for filename in _LONG_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture(scope="session")
def image_assets() -> ImageTestAssets:
    return IMAGE_ASSETS


@pytest.fixture(scope="session")
def video_assets() -> VideoTestAssets:
    return VIDEO_ASSETS


@pytest.fixture(scope="session")
def audio_assets() -> AudioTestAssets:
    return AUDIO_ASSETS


_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_R = TypeVar("_R")


class HfRunner:

    def get_default_device(self):
        from vllm.platforms import current_platform

        return ("cpu"
                if current_platform.is_cpu() else current_platform.device_type)

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
            device = self.device

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
        # Set this to avoid hanging issue
        default_torch_num_threads: Optional[int] = None,
    ) -> None:
        init_ctx = (nullcontext() if default_torch_num_threads is None else
                    set_default_torch_num_threads(default_torch_num_threads))

        with init_ctx:
            self._init(
                model_name=model_name,
                dtype=dtype,
                model_kwargs=model_kwargs,
                trust_remote_code=trust_remote_code,
                is_sentence_transformer=is_sentence_transformer,
                is_cross_encoder=is_cross_encoder,
                skip_tokenizer_init=skip_tokenizer_init,
                auto_cls=auto_cls,
            )

    def _init(
        self,
        model_name: str,
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = True,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
    ) -> None:
        model_name = maybe_model_redirect(model_name)
        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.device = self.get_default_device()
        self.dtype = torch_dtype = _get_and_verify_dtype(
            self.model_name,
            self.config,
            dtype=dtype,
            is_pooling_model=is_sentence_transformer or is_cross_encoder,
        )

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        model_kwargs.setdefault("torch_dtype", torch_dtype)

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args=model_kwargs,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = auto_cls.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )

            # in case some unquantized custom models are not in same dtype
            if (getattr(model, "quantization_method", None) is None
                    and any(p.dtype != self.dtype
                            for p in model.parameters())):
                model = model.to(dtype=self.dtype)

            if (getattr(model, "quantization_method", None) != "bitsandbytes"
                    and len({p.device
                             for p in model.parameters()}) < 2):
                model = model.to(device=self.device)

            self.model = model

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

    def get_inputs(
        self,
        prompts: list[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[Union[BatchFeature, BatchEncoding]]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        all_inputs: list[Union[BatchFeature, BatchEncoding]] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: dict[str, Any] = {
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and (image := images[i]) is not None:
                processor_kwargs["images"] = image
            if videos is not None and (video := videos[i]) is not None:
                processor_kwargs["videos"] = video
            if audios is not None and (audio_inputs := audios[i]) is not None:
                # HACK - not all processors take sampling_rate; we should
                # clean this up in the future.
                if len(audio_inputs) == 2:
                    audio, sr = audio_inputs
                    processor_kwargs["audio"] = audio
                    processor_kwargs["sampling_rate"] = sr
                else:
                    processor_kwargs["audio"] = audio_inputs

            inputs = self.processor(**processor_kwargs)
            if isinstance(inputs, BatchFeature):
                inputs = inputs.to(dtype=self.dtype)

            all_inputs.append(inputs)

        return all_inputs

    def get_prompt_embeddings(self, prompts: list[str]) -> list[torch.Tensor]:
        all_inputs = self.get_inputs(prompts)
        embeddings = []
        for inputs in all_inputs:
            input_ids = self.wrap_device(inputs)["input_ids"]
            embedding = self.model.get_input_embeddings()(input_ids).squeeze(0)
            embeddings.append(embedding)
        return embeddings

    def classify(self, prompts: list[str]) -> list[str]:
        # output is final logits
        all_inputs = self.get_inputs(prompts)
        outputs = []
        problem_type = getattr(self.config, "problem_type", "")

        for inputs in all_inputs:
            output = self.model(**self.wrap_device(inputs))
            if problem_type == "regression":
                logits = output.logits[0].tolist()
            elif problem_type == "multi_label_classification":
                logits = output.logits.sigmoid()[0].tolist()
            else:
                logits = output.logits.softmax(dim=-1)[0].tolist()
            outputs.append(logits)

        return outputs

    def generate(
        self,
        prompts: list[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        outputs: list[tuple[list[list[int]], list[str]]] = []
        for inputs in all_inputs:
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
        prompts: list[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)

        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[tuple[list[list[int]], list[str]]]:
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
                                num_return_sequences=beam_width,
                                images=images,
                                videos=videos,
                                audios=audios)

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
        prompts: list[str],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[list[torch.Tensor]]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        all_logprobs: list[list[torch.Tensor]] = []
        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            seq_logprobs = self._hidden_states_to_seq_logprobs(
                output.hidden_states)
            all_logprobs.append(seq_logprobs)
        return all_logprobs

    def _hidden_states_to_seq_logprobs(
        self,
        hidden_states: tuple[tuple[torch.Tensor, ...], ...],
    ) -> list[torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()

        seq_logprobs: list[torch.Tensor] = []
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                ),
                output_embeddings.weight.t(),
            )
            if getattr(output_embeddings, "bias", None) is not None:
                logits += output_embeddings.bias.unsqueeze(0)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            seq_logprobs.append(logprobs)

        return seq_logprobs

    def _hidden_states_to_logprobs(
        self,
        hidden_states: tuple[tuple[torch.Tensor, ...], ...],
        num_logprobs: Optional[int],
    ) -> tuple[list[dict[int, float]], int]:
        seq_logprobs = self._hidden_states_to_seq_logprobs(hidden_states)
        output_len = len(hidden_states)

        # convert to dict
        seq_logprobs_lst: list[dict[int, float]] = []
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
        prompts: list[str],
        max_tokens: int,
        num_logprobs: Optional[int],
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
    ) -> list[TokensTextLogprobs]:
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

        all_logprobs: list[list[dict[int, float]]] = []
        all_output_ids: list[list[int]] = []
        all_output_strs: list[str] = []

        for inputs in all_inputs:
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

    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def predict(self, prompts: list[list[str]], *args,
                **kwargs) -> torch.Tensor:
        return self.model.predict(prompts,
                                  *args,
                                  convert_to_tensor=True,
                                  **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


class VllmRunner:
    """
    The default value of some arguments have been modified from
    {class}`~vllm.LLM` as follows:

    - `trust_remote_code`: Set to `True` instead of `False` for convenience.
    - `seed`: Set to `0` instead of `None` for test reproducibility.
    - `max_model_len`: Set to `1024` instead of `None` to reduce memory usage.
    - `block_size`: To reduce memory usage, set default to `64` if on XPU
        devices, otherwise default to `16`.
    - `enable_chunked_prefill`: Set to `False` instead of `None` for
      test reproducibility.
    - `enforce_eager`: Set to `False` to test CUDA graph.
    """

    def __init__(
        self,
        model_name: str,
        runner: RunnerOption = "auto",
        convert: ConvertOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = True,
        seed: Optional[int] = 0,
        max_model_len: Optional[int] = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16 if not torch.xpu.is_available() else 64,
        enable_chunked_prefill: Optional[bool] = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        # Set this to avoid hanging issue
        default_torch_num_threads: Optional[int] = None,
        **kwargs,
    ) -> None:
        init_ctx = (nullcontext() if default_torch_num_threads is None else
                    set_default_torch_num_threads(default_torch_num_threads))

        with init_ctx:
            self.llm = LLM(
                model=model_name,
                runner=runner,
                convert=convert,
                tokenizer=tokenizer_name,
                tokenizer_mode=tokenizer_mode,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                seed=seed,
                swap_space=swap_space,
                enforce_eager=enforce_eager,
                disable_log_stats=disable_log_stats,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                block_size=block_size,
                enable_chunked_prefill=enable_chunked_prefill,
                **kwargs,
            )

    def get_inputs(
        self,
        prompts: Union[list[str], list[torch.Tensor], list[int]],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[TextPrompt]:

        if any(x is not None and len(x) != len(prompts)
               for x in [images, videos, audios]):
            raise ValueError(
                "All non-None multimodal inputs must have the same length as "
                "prompts")

        inputs = []
        for i, prompt in enumerate(prompts):
            multi_modal_data = {}
            if images is not None and (image := images[i]) is not None:
                multi_modal_data["image"] = image
            if videos is not None and (video := videos[i]) is not None:
                multi_modal_data["video"] = video
            if audios is not None and (audio := audios[i]) is not None:
                multi_modal_data["audio"] = audio

            text_prompt_kwargs: dict[str, Any] = {
                "multi_modal_data": multi_modal_data or None
            }
            if isinstance(prompt, str):
                text_prompt_kwargs["prompt"] = prompt
            elif isinstance(prompt, list):
                text_prompt_kwargs["prompt_token_ids"] = prompt
            else:
                text_prompt_kwargs["prompt_embeds"] = prompt

            inputs.append(TextPrompt(**text_prompt_kwargs))

        return inputs

    def generate(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.llm.generate(inputs,
                                        sampling_params=sampling_params,
                                        **kwargs)

        outputs: list[tuple[list[list[int]], list[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: list[list[int]] = []
            req_sample_output_strs: list[str] = []
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append((prompt_str or "") + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
        req_outputs: list[RequestOutput],
    ) -> list[TokensTextLogprobsPromptLogprobs]:
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.llm.generate(inputs,
                                        sampling_params=sampling_params,
                                        **kwargs)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
        prompts: Union[list[str], list[torch.Tensor]],
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
    ) -> list[tuple[list[int], str]]:
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
        prompts: list[str],
        max_tokens: int,
        num_logprobs: Optional[int],
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop)

        return self.generate_w_logprobs(prompts,
                                        greedy_logprobs_params,
                                        images=images,
                                        audios=audios,
                                        videos=videos,
                                        **kwargs)

    def generate_prompt_perplexity(self, prompts: list[str]) -> list[float]:
        """
        Return the perplexity score associated with generating the prompts

        :param prompts: list of prompts to score
        :return: perplexity score of each prompt
        """
        outputs = self.generate_greedy_logprobs(prompts,
                                                max_tokens=1,
                                                num_logprobs=None,
                                                num_prompt_logprobs=0)

        perplexities = []
        for output in outputs:
            output = cast(TokensTextLogprobsPromptLogprobs, output)
            token_datas = cast(list[Optional[dict[int, Logprob]]], output[3])
            assert token_datas[0] is None
            token_log_probs = []
            for token_data in token_datas[1:]:
                assert token_data is not None
                assert len(token_data) == 1
                token_log_prob = list(token_data.values())[0].logprob
                token_log_probs.append(token_log_prob)

            perplexity = math.exp(-sum(token_log_probs) / len(token_log_probs))
            perplexities.append(perplexity)

        return perplexities

    def generate_beam_search(
        self,
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        concurrency_limit: Optional[int] = None,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        outputs = self.llm.beam_search(inputs,
                                       BeamSearchParams(beam_width=beam_width,
                                                        max_tokens=max_tokens),
                                       concurrency_limit=concurrency_limit)
        returned_outputs = []
        for output in outputs:
            token_ids = [x.tokens for x in output.sequences]
            texts = [x.text for x in output.sequences]
            returned_outputs.append((token_ids, texts))
        return returned_outputs

    def classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def embed(self,
              prompts: list[str],
              images: Optional[PromptImageInput] = None,
              videos: Optional[PromptVideoInput] = None,
              audios: Optional[PromptAudioInput] = None,
              *args,
              **kwargs) -> list[list[float]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.llm.embed(inputs, *args, **kwargs)
        return [req_output.outputs.embedding for req_output in req_outputs]

    def encode(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.encode(prompts)
        return [req_output.outputs.data for req_output in req_outputs]

    def reward(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.llm.reward(prompts)
        return [req_output.outputs.data for req_output in req_outputs]

    def score(
        self,
        text_1: Union[str, list[str]],
        text_2: Union[str, list[str]],
        *args,
        **kwargs,
    ) -> list[float]:
        req_outputs = self.llm.score(text_1, text_2, *args, **kwargs)
        return [req_output.outputs.score for req_output in req_outputs]

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.llm.apply_model(func)

    def get_llm(self) -> LLM:
        return self.llm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.llm
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


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

    from vllm.platforms import current_platform
    return current_platform.device_count()


temp_dir = tempfile.gettempdir()
_dummy_opt_path = os.path.join(temp_dir, "dummy_opt")
_dummy_llava_path = os.path.join(temp_dir, "dummy_llava")
_dummy_gemma2_embedding_path = os.path.join(temp_dir, "dummy_gemma2_embedding")


@pytest.fixture
def dummy_opt_path():
    json_path = os.path.join(_dummy_opt_path, "config.json")
    if not os.path.exists(_dummy_opt_path):
        snapshot_download(repo_id="facebook/opt-125m",
                          local_dir=_dummy_opt_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyOPTForCausalLM"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_opt_path


@pytest.fixture
def dummy_llava_path():
    json_path = os.path.join(_dummy_llava_path, "config.json")
    if not os.path.exists(_dummy_llava_path):
        snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf",
                          local_dir=_dummy_llava_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyLlava"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_llava_path


@pytest.fixture
def dummy_gemma2_embedding_path():
    json_path = os.path.join(_dummy_gemma2_embedding_path, "config.json")
    if not os.path.exists(_dummy_gemma2_embedding_path):
        snapshot_download(repo_id="BAAI/bge-multilingual-gemma2",
                          local_dir=_dummy_gemma2_embedding_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyGemma2Embedding"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_gemma2_embedding_path


# Add the flag `--optional` to allow run tests
# that are marked with @pytest.mark.optional
def pytest_addoption(parser):
    parser.addoption("--optional",
                     action="store_true",
                     default=False,
                     help="run optional test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional"):
        # --optional given in cli: do not skip optional tests
        return
    skip_optional = pytest.mark.skip(reason="need --optional option to run")
    for item in items:
        if "optional" in item.keywords:
            item.add_marker(skip_optional)


@pytest.fixture(scope="session")
def cli_config_file():
    """Return the path to the CLI config file."""
    return os.path.join(_TEST_DIR, "config", "test_config.yaml")


@pytest.fixture(scope="session")
def cli_config_file_with_model():
    """Return the path to the CLI config file with model."""
    return os.path.join(_TEST_DIR, "config", "test_config_with_model.yaml")


class AssetHandler(http.server.BaseHTTPRequestHandler):
    # _IMAGE_CACHE : Dict[str, bytes] = {}

    def log_message(self, *args, **kwargs):
        pass

    def do_GET(self):
        # Accepts paths like: /1280px-Venn_diagram_rgb.jpg
        filename = self.path.lstrip("/")
        if not filename or "." not in filename:
            self.send_error(404, "Missing filename (expected /<name>.<ext>)")
            return

        base, ext = filename.rsplit(".", 1)
        ext = ext.lower()

        if ext not in ["jpg", "png"]:
            self.send_error(404, f"Unsupported extension: .{ext}")
            return

        try:
            data = ImageAsset(base).read_bytes(ext=ext)
        except Exception as e:
            self.send_error(500, f"Failed to load asset: {ext} {base} {e} ")
            return

        ctype, _ = mimetypes.guess_type(filename)
        if ctype is None:
            ctype = {"jpg": "image/jpg", "png": "image/png"}[ext]
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalAssetServer:

    address: str
    port: int
    server: Optional[http.server.ThreadingHTTPServer]
    thread: Optional[threading.Thread]

    def __init__(self, address: str = "127.0.0.1") -> None:
        self.address = address
        self.port = -1
        self.server = None
        self.thread = None

    def __enter__(self):
        self.port = _find_free_port()
        self.server = http.server.ThreadingHTTPServer(
            (self.address, self.port), AssetHandler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.server:
            self.server.shutdown()
            del self.server

        if self.thread:
            self.thread.join()
            del self.thread

        if exc_type is None:
            return None

        return False

    @property
    def base_url(self) -> str:
        assert self.port is not None
        return f"http://{self.address}:{self.port}"

    def url_for(self, name: str) -> str:
        """e.g., name='RGBA_comp.png' -> 'http://127.0.0.1:PORT/RGBA_comp.png'"""
        return f"{self.base_url}/{name}"

    def get_image_asset(self, name: str) -> Image.Image:
        return fetch_image(self.url_for(name))


@pytest.fixture(scope="session")
def local_asset_server() -> Generator[LocalAssetServer, None, None]:
    """
    Starts a thread based HTTP server bound to 127.0.0.1 on a random free port. 
    The server currently servers images at:
    http://127.0.0.1:<port>/<name>.<ext>
    """
    with LocalAssetServer() as srv:
        yield srv


@pytest.fixture
def image_url(request, local_asset_server) -> str:
    # request.param is one of the IMAGE_ASSETS filenames
    name = request.param
    return local_asset_server.url_for(name)


@pytest.fixture
def image_urls(request, local_asset_server) -> list[str]:
    """Indirect fixture: takes a list of names, returns list of full URLs."""
    names: list[str] = request.param
    return [local_asset_server.url_for(name) for name in names]
