from typing import List, Optional, Tuple, Type, overload

import pytest
from transformers import (AutoConfig, AutoModelForVision2Seq, AutoTokenizer,
                          BatchEncoding)

from vllm.attention.selector import (_Backend, _cached_get_attn_backend,
                                     global_force_attn_backend_context_manager)
from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs

from ....conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
                          _ImageAssets)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

_LIMIT_IMAGE_PER_PROMPT = 3

LIST_ENC_DEC_SUPPORTED_BACKENDS = [_Backend.XFORMERS, _Backend.FLASH_ATTN]

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|image|><|begin_of_text|>The meaning of the image is",
    "cherry_blossom":
    "<|image|><|begin_of_text|>The city is",
})

text_only_prompts = [
    "The color of the sky is blue but sometimes it can also be",
]

models = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]],
                      model: str):
    """Sanitize vllm output to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    hf_output_str = output_str
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def _get_inputs(
    image_assets: _ImageAssets,
    *,
    size_factors: Optional[List[float]] = None,
    sizes: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[List[str], PromptImageInput]]:
    images = [asset.pil_image for asset in image_assets]

    if size_factors is not None:
        inputs_per_image = [(
            [prompt for _ in size_factors],
            [rescale_image_size(image, factor) for factor in size_factors],
        ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]
    elif sizes is not None:
        inputs_per_image = [(
            [
                prompt if size is not None else text_only_prompts[0]
                for size in sizes
            ],
            [
                image.resize(size) if size is not None else None
                for size in sizes
            ],
        ) for image, prompt in zip(images, HF_IMAGE_PROMPTS)]
        if len(sizes) == 0:
            inputs_per_image.append(
                (text_only_prompts, [None] * len(text_only_prompts)))
    else:
        raise ValueError("You must provide either `size_factors` or `sizes`")

    return inputs_per_image


@overload
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


@overload
def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    sizes: List[Tuple[int, int]],
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    ...


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model: str,
    *,
    size_factors: Optional[List[float]] = None,
    sizes: Optional[List[Tuple[int, int]]] = None,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    _run_test(
        hf_runner,
        vllm_runner,
        _get_inputs(image_assets, size_factors=size_factors, sizes=sizes),
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
    )


def _run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], PromptImageInput]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test are from IMAGE_ASSETS.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalDataDict objects 
    and corresponding MultiModalConfig as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     dtype=dtype,
                     max_model_len=4096,
                     max_num_seqs=2,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     limit_mm_per_prompt={"image": _LIMIT_IMAGE_PER_PROMPT
                                          }) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    def process(hf_inputs: BatchEncoding, **kwargs):
        return hf_inputs

    with hf_runner(model,
                   dtype=dtype,
                   model_kwargs={"device_map": "auto"},
                   postprocess_inputs=process,
                   auto_cls=AutoModelForVision2Seq) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images)
            for prompts, images in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, model)
                for vllm_output in vllm_outputs
            ],
            name_0="hf",
            name_1="vllm",
        )


@pytest.fixture(autouse=True)
def clear_cache():
    """Fixture to clear backend cache before each test."""
    _cached_get_attn_backend.cache_clear()  # Clear the cache
    yield  # This allows the test to run


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize(
    "sizes",
    [
        # Text only
        [],
        # Single-size
        [(512, 512)],
        # Single-size, batched
        [(512, 512), (512, 512), (512, 512)],
        # Multi-size, batched
        [(512, 512), (1024, 512), (1536, 512), (2048, 512), (512, 1024),
         (1024, 1024), (512, 1536), (512, 2028)],
        # Multi-size, batched, including text only
        [(512, 512), (1024, 512), (1536, 512), (2048, 512), (512, 1024),
         (1024, 1024), (512, 1536), (512, 2028), None],
        # mllama has 8 possible aspect ratios, carefully set the sizes
        # to cover all of them
    ])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
def test_models_single_leading_image(hf_runner, vllm_runner, image_assets,
                                     model, sizes, dtype, max_tokens,
                                     num_logprobs,
                                     attn_backend: _Backend) -> None:
    with global_force_attn_backend_context_manager(attn_backend):
        if attn_backend == _Backend.FLASH_ATTN:
            # Flash Attention works only with bfloat16 data-type
            dtype = 'bfloat16'
        run_test(
            hf_runner,
            vllm_runner,
            image_assets,
            model,
            sizes=sizes,
            dtype=dtype,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            tensor_parallel_size=1,
        )


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
def test_models_multi_leading_images(hf_runner, vllm_runner, image_assets,
                                     model, dtype, max_tokens, num_logprobs,
                                     attn_backend: _Backend) -> None:

    stop_sign = image_assets[0].pil_image
    cherry_blossom = image_assets[1].pil_image

    inputs = [(
        [
            "<|image|><|image|><|begin_of_text|>Describe 2 images.",  # noqa: E501
            "<|image|><|image|><|begin_of_text|>Describe 2 images.",  # noqa: E501
            "<|image|><|image|><|image|><|begin_of_text|>Describe 3 images.",  # noqa: E501
        ],
        [
            [stop_sign, cherry_blossom],
            # Images with different sizes.
            [
                stop_sign.resize((512, 512)),
                stop_sign,
            ],
            [
                stop_sign,
                stop_sign.resize((512, 1536)),
                cherry_blossom.resize((512, 1024)),
            ],
        ])]
    with global_force_attn_backend_context_manager(attn_backend):
        if attn_backend == _Backend.FLASH_ATTN:
            # Flash Attention works only with bfloat16 data-type
            dtype = 'bfloat16'
        _run_test(
            hf_runner,
            vllm_runner,
            inputs,
            model,
            dtype=dtype,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            tensor_parallel_size=1,
        )


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
def test_models_interleaved_images(hf_runner, vllm_runner, image_assets, model,
                                   dtype, max_tokens, num_logprobs,
                                   attn_backend: _Backend) -> None:

    stop_sign = image_assets[0].pil_image
    cherry_blossom = image_assets[1].pil_image

    inputs = [(
        [
            "<|begin_of_text|>The content of the image <|image|> is",  # noqa: E501
            "<|begin_of_text|>Between the first image <|image|> and the second image<|image|>, "  # noqa: E501
            "which is a stop sign and which is a cherry blossom?",  # noqa: E501
        ],
        [
            [stop_sign],
            [stop_sign, cherry_blossom],
        ])]
    with global_force_attn_backend_context_manager(attn_backend):
        if attn_backend == _Backend.FLASH_ATTN:
            # Flash Attention works only with bfloat16 data-type
            dtype = 'bfloat16'
        _run_test(
            hf_runner,
            vllm_runner,
            inputs,
            model,
            dtype=dtype,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            tensor_parallel_size=1,
        )
