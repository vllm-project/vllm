import contextlib
import gc
import os
from collections import UserList
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (Any, Dict, List, Literal, Optional, Tuple, TypedDict,
                    TypeVar)

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer, BatchEncoding)

from vllm import LLM, SamplingParams
from vllm.config import TokenizerPoolConfig, VisionLanguageConfig
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from vllm.inputs import TextPrompt
from vllm.logger import init_logger
from vllm.multimodal import MultiModalData
from vllm.multimodal.image import ImageFeatureData, ImagePixelData
from vllm.sequence import SampleLogprobs
from vllm.utils import cuda_device_count_stateless, is_cpu

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]

_IMAGE_DIR = Path(_TEST_DIR) / "images"
"""You can use `.buildkite/download-images.sh` to download the assets."""


def _read_prompts(filename: str) -> List[str]:
    with open(filename, "r") as f:
        prompts = f.readlines()
        return prompts


@dataclass(frozen=True)
class ImageAsset:
    name: Literal["stop_sign", "cherry_blossom"]

    @cached_property
    def pixel_values(self) -> torch.Tensor:
        return torch.load(_IMAGE_DIR / f"{self.name}_pixel_values.pt")

    @cached_property
    def image_features(self) -> torch.Tensor:
        return torch.load(_IMAGE_DIR / f"{self.name}_image_features.pt")

    @cached_property
    def pil_image(self) -> Image.Image:
        return Image.open(_IMAGE_DIR / f"{self.name}.jpg")

    def for_hf(self) -> Image.Image:
        return self.pil_image

    def for_vllm(self, vision_config: VisionLanguageConfig) -> MultiModalData:
        image_input_type = vision_config.image_input_type
        ImageInputType = VisionLanguageConfig.ImageInputType

        if image_input_type == ImageInputType.IMAGE_FEATURES:
            return ImageFeatureData(self.image_features)
        if image_input_type == ImageInputType.PIXEL_VALUES:
            return ImagePixelData(self.pil_image)

        raise NotImplementedError


class _ImageAssetPrompts(TypedDict):
    stop_sign: str
    cherry_blossom: str


class _ImageAssets(UserList[ImageAsset]):

    def __init__(self) -> None:
        super().__init__(
            [ImageAsset("stop_sign"),
             ImageAsset("cherry_blossom")])

    def prompts(self, prompts: _ImageAssetPrompts) -> List[str]:
        """
        Convenience method to define the prompt for each test image.

        The order of the returned prompts matches the order of the
        assets when iterating through this object.
        """
        return [prompts["stop_sign"], prompts["cherry_blossom"]]


IMAGE_ASSETS = _ImageAssets()
"""Singleton instance of :class:`_ImageAssets`."""


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


@pytest.fixture
def example_long_prompts() -> List[str]:
    prompts = []
    for filename in _LONG_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture(scope="session")
def image_assets() -> _ImageAssets:
    return IMAGE_ASSETS


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding)


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
    ) -> None:
        assert dtype in _STR_DTYPE_TO_TORCH_DTYPE
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

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
        **kwargs,
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
    ) -> List[List[torch.Tensor]]:
        all_logprobs = []
        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                self.wrap_device(input_ids),
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
                logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                seq_logprobs.append(logprobs)
            all_logprobs.append(seq_logprobs)
        return all_logprobs

    def generate_greedy_logprobs_limit(
        self,
        prompts: List[str],
        max_tokens: int,
        num_logprobs: int,
    ) -> List[Tuple[List[int], str, List[Dict[int, float]]]]:
        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for prompt in prompts:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                self.wrap_device(input_ids),
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            seq_logprobs: List[torch.Tensor] = []
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
        enforce_eager: bool = False,
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
        images: Optional[List[MultiModalData]] = None,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        if images is not None:
            assert len(prompts) == len(images)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["multi_modal_data"] = image

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
                output_ids = sample.token_ids
                req_sample_output_ids.append(prompt_ids + output_ids)
                req_sample_output_strs.append(prompt_str + output_str)
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    def generate_w_logprobs(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
        assert sampling_params.logprobs is not None

        req_outputs = self.model.generate(prompts,
                                          sampling_params=sampling_params)
        outputs: List[Tuple[List[int], str, Optional[SampleLogprobs]]] = []
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
        images: Optional[List[MultiModalData]] = None,
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
    ) -> List[Tuple[List[int], str, Optional[SampleLogprobs]]]:
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
