# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Type, overload

import pytest
import torch
from transformers import (AutoConfig, AutoModelForVision2Seq, AutoTokenizer,
                          BatchEncoding)

from vllm import LLM, SamplingParams
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.selector import (_Backend, _cached_get_attn_backend,
                                     global_force_attn_backend_context_manager)
from vllm.model_executor.models.mllama import MllamaForConditionalGeneration
from vllm.multimodal.image import rescale_image_size
from vllm.sequence import SampleLogprobs

from ....conftest import (IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner,
                          _ImageAssets)
from ....utils import large_gpu_test
from ...utils import check_logprobs_close

_LIMIT_IMAGE_PER_PROMPT = 3
MLLAMA_IMAGE_TOKEN_ID = 128256

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

# Indices for inputs
TEXT_ONLY = '0'
IMAGE_AT_BEG = '1'
IMAGE_AT_MIDDLE = '2'
TWO_IMAGES = '3'

# Input tokenized
prompt_data = {
    # Tell me a story
    TEXT_ONLY: [41551, 757, 264, 3446],
    # <|image|> What's the content of this image
    IMAGE_AT_BEG:
    [MLLAMA_IMAGE_TOKEN_ID, 3639, 596, 279, 2262, 315, 420, 2217, 220],
    # Hello <|image|>What' the content of this image
    IMAGE_AT_MIDDLE:
    [9906, 220, MLLAMA_IMAGE_TOKEN_ID, 3923, 6, 279, 2262, 315, 420, 2217],
    #<|image|>Is there a duck in this image?<|image|>What's the animal in this image? # noqa: E501
    TWO_IMAGES: [
        MLLAMA_IMAGE_TOKEN_ID, 3957, 1070, 264, 37085, 304, 420, 2217, 30,
        MLLAMA_IMAGE_TOKEN_ID, 3923, 596, 279, 10065, 304, 420, 2217, 30
    ]
}


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


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [32])
def test_explicit_implicit_prompt(
    image_assets: _ImageAssets,
    model: str,
    dtype: str,
    max_tokens: int,
):
    stop_sign = image_assets[0].pil_image
    # yapf: disable
    prompts = [
        # explicit prompt
        {
            "encoder_prompt": {
                "prompt": "<|image|>",
                "multi_modal_data": {"image": stop_sign},
            },
            "decoder_prompt": {
                "prompt_token_ids": [128000, 791, 2262, 315, 279, 2217, 220, 128256, 374],  # noqa: E501
            }
        },
        {
            "encoder_prompt": "Not <|image|>",
            "decoder_prompt": "The color of the sky is blue but sometimes it can also be",  # noqa: E501
        },
        # implicit prompt
        {
            "prompt": "<|begin_of_text|>The content of the image <|image|> is", # noqa: E501
            "multi_modal_data": {"image": stop_sign},
        },
        {
            "prompt": "The color of the sky is blue but sometimes it can also be",  # noqa: E501
        },
    ]
    # yapf: enable
    llm = LLM(
        model=model,
        dtype=dtype,
        max_model_len=4096,
        max_num_seqs=2,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    n_prompts = len(prompts)
    explicit_outputs = outputs[:n_prompts // 2]
    implicit_outputs = outputs[n_prompts // 2:]
    for exp_output, imp_output in zip(explicit_outputs, implicit_outputs):
        assert exp_output.outputs[0].text == imp_output.outputs[0].text


@large_gpu_test(min_gb=48)
@pytest.mark.core_model
@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
def test_regression(vllm_runner, image_assets, model, dtype, max_tokens,
                    num_logprobs, attn_backend: _Backend) -> None:

    stop_sign = image_assets[0].pil_image

    with global_force_attn_backend_context_manager(attn_backend), vllm_runner(
            model,
            dtype=dtype,
            max_model_len=4096,
            max_num_seqs=2,
            tensor_parallel_size=1,
            enforce_eager=True,
            limit_mm_per_prompt={"image":
                                 _LIMIT_IMAGE_PER_PROMPT}) as vllm_model:

        # Regression tests for https://github.com/vllm-project/vllm/issues/10648

        # Number of groups of image tokens is greater than the number of images
        # provided (the whitespace between the tags is necessary)
        prompt = "<|begin_of_text|><|image|> <|image|> Compare the two images"  # noqa: E501
        image = stop_sign
        with pytest.raises(ValueError):
            vllm_model.generate_greedy_logprobs([prompt],
                                                max_tokens,
                                                num_logprobs,
                                                images=[image])

        # Batch of a text-only and image request that requires cross-attention
        prompts = [
            "What is the capital of spain?",
            "Text before the image...<|image|>What is in the image?",  # noqa: E501
        ]
        images = [
            None,
            [stop_sign],
        ]
        vllm_model.generate_greedy_logprobs(prompts,
                                            max_tokens,
                                            num_logprobs,
                                            images=images)

        # Test the reverse order too for good measure
        prompts = [
            "<|begin_of_text|>Text before the image...<|image|>What is in the image?",  # noqa: E501
            "<|begin_of_text|>Hello!",
        ]
        images = [
            [stop_sign],
            None,
        ]
        vllm_model.generate_greedy_logprobs(prompts,
                                            max_tokens,
                                            num_logprobs,
                                            images=images)


class DummyModel:
    image_token_id = MLLAMA_IMAGE_TOKEN_ID


@pytest.mark.core_model
@pytest.mark.parametrize(
    "input_indices_and_output",
    # inputs, (cross_attention_mask, kv_range_for_decode)
    [([TEXT_ONLY], (None, None)), ([IMAGE_AT_BEG], (None, None)),
     ([TEXT_ONLY, IMAGE_AT_BEG], (None, None)),
     ([IMAGE_AT_MIDDLE], ((10, 12), [[0, 6]])),
     ([TEXT_ONLY, IMAGE_AT_MIDDLE], ((14, 12), [[0, 6]])),
     ([TEXT_ONLY, IMAGE_AT_BEG, IMAGE_AT_MIDDLE],
      ((23, 24), [[0, 6], [6, 12]])),
     ([IMAGE_AT_MIDDLE, TEXT_ONLY], ((14, 12), [[0, 6]])),
     ([TWO_IMAGES], ((18, 12), [[6, 12]])),
     ([TEXT_ONLY, TWO_IMAGES], ((22, 12), [[6, 12]]))])
def test_get_cross_attention_mask(input_indices_and_output) -> None:

    input_indices, expected_output = input_indices_and_output

    sequences = [torch.tensor(prompt_data[i]) for i in input_indices]
    num_tiles = [[2, 2] if i != TEXT_ONLY else [] for i in input_indices
                 if i != TEXT_ONLY]
    input = torch.cat(sequences)

    seq_lens = [len(s) for s in sequences]

    attn_data = FlashAttentionMetadata(
        seq_lens=seq_lens,
        # Dummy values
        enable_kv_scales_calculation=False,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=0,
        slot_mapping=0,
        multi_modal_placeholder_index_maps=None,
        seq_lens_tensor=0,
        max_prefill_seq_len=0,
        max_decode_seq_len=0,
        context_lens_tensor=None,
        block_tables=None,
        use_cuda_graph=False,
    )

    dummy = DummyModel()

    cross_attention_mask, kv_range_for_decode = MllamaForConditionalGeneration\
        .get_cross_attention_mask(dummy,
                                  input,
                                  attn_data,
                                  num_tiles=num_tiles,
                                  num_tokens_per_tile=3,
                                  dtype=torch.bfloat16)

    expected_cross_attention_mask, expected_kv_range_for_decode = \
        expected_output

    assert kv_range_for_decode == expected_kv_range_for_decode
    if expected_cross_attention_mask is not None:
        assert cross_attention_mask is not None
        assert cross_attention_mask.shape == expected_cross_attention_mask
    else:
        assert cross_attention_mask is None


@pytest.mark.core_model
@pytest.mark.parametrize(
    "input_indices",
    [[TEXT_ONLY], [IMAGE_AT_BEG], [TEXT_ONLY, IMAGE_AT_BEG], [IMAGE_AT_MIDDLE],
     [TEXT_ONLY, IMAGE_AT_MIDDLE], [TEXT_ONLY, IMAGE_AT_BEG, IMAGE_AT_MIDDLE],
     [IMAGE_AT_MIDDLE, TEXT_ONLY], [TWO_IMAGES], [TEXT_ONLY, TWO_IMAGES]])
def test_get_full_text_row_masked_out_mask(input_indices) -> None:

    sequences = [torch.tensor(prompt_data[i]) for i in input_indices]

    seq_lens = [len(s) for s in sequences]

    num_prefill_tokens = sum(seq_lens)

    # TEXT_ONLY is zero, so it will be masked out,
    # other instances should not be.
    encoder_seq_lens = [int(i) for i in input_indices]

    attn_data = FlashAttentionMetadata(
        seq_lens=seq_lens,
        encoder_seq_lens=encoder_seq_lens,
        num_prefill_tokens=num_prefill_tokens,
        # Dummy values
        enable_kv_scales_calculation=False,
        num_prefills=0,
        num_decode_tokens=0,
        slot_mapping=0,
        multi_modal_placeholder_index_maps=None,
        seq_lens_tensor=0,
        max_prefill_seq_len=0,
        max_decode_seq_len=0,
        context_lens_tensor=None,
        block_tables=None,
        use_cuda_graph=False,
    )

    dummy = DummyModel()

    full_text_row_masked_out_mask = MllamaForConditionalGeneration\
        .get_full_text_row_masked_out_mask(dummy,
                                  attn_data,
                                  torch.get_default_device())

    full_text_row_masked_out_mask = full_text_row_masked_out_mask.squeeze()
    full_text_row_masked_out_mask = full_text_row_masked_out_mask.tolist()

    idx = 0
    assert len(full_text_row_masked_out_mask) == num_prefill_tokens
    for i, seq_len in enumerate(seq_lens):
        must_be_masked = input_indices[i] != TEXT_ONLY
        for _ in range(seq_len):
            assert full_text_row_masked_out_mask[idx] == must_be_masked, \
                f"full_text_row_masked_out_mask[{idx}] must be " \
                f"'{must_be_masked}' "
            idx += 1
