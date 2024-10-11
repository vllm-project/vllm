"""Common tests for testing .generate() functionality for single / multiple
image support for different VLMs in vLLM.
"""
import os
from pathlib import PosixPath
from typing import (Any, Callable, Dict, List, Optional, Tuple,
                    Type, Union)

import pytest
from PIL.Image import Image
import torch
from transformers import (AutoModelForVision2Seq, AutoTokenizer, BatchEncoding)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs
from vllm.utils import identity, is_cpu, is_hip

from ....conftest import HfRunner, VllmRunner, _ImageAssets
from ...utils import check_outputs_equal
import utils as vlm_utils
from vlm_test_types import (
    VLMTestInfo,
    VlmTestType,
    SINGLE_IMAGE_BASE_PROMPTS,
    MULTI_IMAGE_BASE_PROMPT,
)

# This hack is needed for phi3v & paligemma models
# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if is_hip():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

### Test configuration for specific models;
# NOTE: the key in the dict below is not mostly used as an identifier;
# it will be first in all of the expanded parametrizations, so it will
# tell you which test configuration failed.

# yapf: disable
VLM_TEST_SETTINGS = {
    "blip2": VLMTestInfo(
        models="Salesforce/blip2-opt-2.7b",
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        vllm_output_post_proc=vlm_utils.blip2_vllm_to_hf_output,
        auto_cls=AutoModelForVision2Seq
    ),
    "fuyu": VLMTestInfo(
        models="adept/fuyu-8b",
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        vllm_output_post_proc=vlm_utils.fuyu_vllm_to_hf_output,
        dtype="bfloat16" if is_cpu() else "half",
        num_logprobs=10,
        max_model_len=2048,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        image_size_factors=((), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15))
    ),
    "chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        dtype="bfloat16",
        postprocess_inputs=vlm_utils.get_key_type_post_processor(
            "pixel_values", 
            "bfloat16"
        ),
        max_model_len=4096,
        max_tokens=8,
        auto_cls=AutoModelForVision2Seq,
        # For chameleon, we only compare the sequences
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
    ),
    # Only embedding test has been verified so far
    "llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        vllm_output_post_proc=vlm_utils.llava_vllm_to_hf_output,
        auto_cls=AutoModelForVision2Seq,
        convert_assets_to_embeddings=vlm_utils.get_llava_embeddings,
    ),
    "minicpmv": VLMTestInfo(
        models="openbmb/MiniCPM-Llama3-V-2_5",
        supports_multi_image=True,
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        postprocess_inputs=vlm_utils.wrap_inputs_post_processor,
        max_model_len=4096,
        max_num_seqs=2,
        hf_output_post_proc=vlm_utils.minicmpv_trunc_hf_output,
        get_stop_token_ids=lambda tok: [tok.eos_id, tok.eot_id]
    ),
    "phi3v": VLMTestInfo(
        models="microsoft/Phi-3.5-vision-instruct",
        supports_multi_image=True,
        img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n", # noqa: E501
        vllm_output_post_proc=vlm_utils.phi3v_vllm_to_hf_output,
        num_logprobs=10,
        max_model_len=4096,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        # use eager mode for hf runner, since phi3v didn't work with flash_attn
        model_kwargs={"_attn_implementation": "eager"},
    ),
    "qwen": VLMTestInfo(
        models="Qwen/Qwen-VL",
        supports_multi_image=True,
        vllm_output_post_proc=vlm_utils.qwen_vllm_to_hf_output,
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        prompt_formatter=identity,
        max_model_len=1024,
        max_num_seqs=2,
        prompt_path_encoder=vlm_utils.qwen_prompt_path_encoder,
    ),

    ## Tests above this point have been validated to align with current tests 
    "paligemma": VLMTestInfo(
        models="google/paligemma-3b-mix-224",
        dtype="half" if is_hip() else ["half", "float"],
        # <image> is the paligemma placeholder, which is contained in the
        # default; careful not to pass it in the prompt, or it'll be a mismatch
        img_idx_to_prompt = lambda idx: "",
        prompt_formatter=identity,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.paligemma_vllm_to_hf_output,
        skip=False
    )

    # "intern_vl": VLMTestInfo(
    #     models=["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B"],
    #     supports_multi_image=True,
    #     prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
    #     dtype="bfloat16" if is_cpu() else "half",
    #     num_logprobs=10,
    #     max_model_len=4096,
    # ),
}
# yapf: enable

### Test wrappers
# Wrappers around the core test running func for:
# - single image
# - multi-image
# - image embeddings
# - video [TODO]
# - single image quantized comparisons [TODO]
# All wrappers (except single image) have a filter for dropping
# models that don't have applicable tests, and expanding the
# relevant VLMTestInfo object into a combination that can be
# consumed by parametrize()
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_factors",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS, test_type=VlmTestType.IMAGE))
def test_single_image_generation(tmp_path: PosixPath, model_type: str,
                                 model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 size_factors: Tuple[float], hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompts = vlm_utils.get_model_prompts(
        SINGLE_IMAGE_BASE_PROMPTS,
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter
    )

    # For models that require a local path / URL encoded in the image; export
    # assets and encode into tmp_path for this test. This should be avoided
    # where possible (currently needed for Qwen-VL).
    if test_info.prompt_path_encoder is not None:
        model_prompts = [
            test_info.prompt_path_encoder(tmp_path, prompt, [asset])
            for prompt, asset in zip(model_prompts, image_assets)
        ]

    images = [asset.pil_image for asset in image_assets]
    assert len(images) == len(model_prompts)

    # For every image / prompt pair, get a pair containing two lists of
    # length size_factors, where the first contains duplicates of the model
    # prompt [str], and the second contains copies of the image after being
    # scaled by one of the size factors.
    #
    # NOTE: rescaling preserves the image aspect ratio.
    inputs = [(
        [prompt for _ in size_factors],
        [rescale_image_size(image, factor) for factor in size_factors],
    ) for image, prompt in zip(images, model_prompts)]

    run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=test_info.tensor_parallel_size,
        enforce_eager=test_info.enforce_eager,
        max_model_len=test_info.max_model_len,
        max_num_seqs=test_info.max_num_seqs,
        hf_output_post_proc=test_info.hf_output_post_proc,
        vllm_output_post_proc=test_info.vllm_output_post_proc,
        auto_cls=test_info.auto_cls,
        use_tokenizer_eos=test_info.use_tokenizer_eos,
        postprocess_inputs=test_info.postprocess_inputs,
        comparator=test_info.comparator,
        get_stop_token_ids=test_info.get_stop_token_ids,
        limit_mm_per_prompt={"image": 1},
        model_kwargs=test_info.model_kwargs,
    )

def test_video_generation():
    raise NotImplementedError("Video model test wrapper not implemented")

def test_quantized_models():
    raise NotImplementedError("Quantized model test wrapper not implemented")

@pytest.mark.skipif(not any(test.convert_assets_to_embeddings
                            for test in VLM_TEST_SETTINGS.values()),
                    reason="No models with image embedding tests are enabled.")
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_factors",
    vlm_utils.get_parametrized_options(
        VLM_TEST_SETTINGS, test_type=VlmTestType.EMBEDDING
    )
)
def test_embedding_generation(model_type: str, model: str, max_tokens: int,
                              num_logprobs: int, dtype: str,
                              size_factors: Tuple[float], hf_runner,
                              vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompts = vlm_utils.get_model_prompts(
        SINGLE_IMAGE_BASE_PROMPTS,
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter
    )

    images = [asset.pil_image for asset in image_assets]
    embeddings = test_info.convert_assets_to_embeddings(image_assets)
    assert len(images) == len(model_prompts)

    # NOTE: Not doing any rescaling here at the moment. All size factors
    # for embeddings are 1.0 & just dictate the number of images here.
    inputs = [(
        [prompt for _ in size_factors],
        [image for _ in size_factors],
    ) for image, prompt in zip(images, model_prompts)]

    embeddings = [(
        [prompt for _ in size_factors],
        [image for _ in size_factors],
    ) for image, prompt in zip(embeddings, model_prompts)]

    run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=test_info.tensor_parallel_size,
        enforce_eager=test_info.enforce_eager,
        max_model_len=test_info.max_model_len,
        max_num_seqs=test_info.max_num_seqs,
        hf_output_post_proc=test_info.hf_output_post_proc,
        vllm_output_post_proc=test_info.vllm_output_post_proc,
        auto_cls=test_info.auto_cls,
        use_tokenizer_eos=test_info.use_tokenizer_eos,
        postprocess_inputs=test_info.postprocess_inputs,
        comparator=test_info.comparator,
        get_stop_token_ids=test_info.get_stop_token_ids,
        vllm_embeddings=embeddings,
        limit_mm_per_prompt={"image": 1},
        model_kwargs=test_info.model_kwargs,
    )


@pytest.mark.skipif(not any(test.supports_multi_image
                            for test in VLM_TEST_SETTINGS.values()),
                    reason="No models with multi-image tests are enabled.")
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_factors",
    vlm_utils.get_parametrized_options(
        VLM_TEST_SETTINGS,
        test_type=VlmTestType.MULTI_IMAGE
    )
)
def test_multi_image_generation(tmp_path: PosixPath,model_type: str, model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 size_factors: Tuple[float], hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompt = vlm_utils.get_model_prompts(
        [MULTI_IMAGE_BASE_PROMPT],
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter
    )[0]

    if test_info.prompt_path_encoder is not None:
        model_prompt = test_info.prompt_path_encoder(tmp_path, model_prompt, image_assets)

    images = [asset.pil_image for asset in image_assets]

    # This is similar to the single image case, but we rescale each of the
    # images in the multi-image prompt; currently we only have one model prompt
    inputs = [([model_prompt for _ in size_factors],
               [[rescale_image_size(image, factor) for image in images]
                for factor in size_factors])]

    run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=test_info.tensor_parallel_size,
        enforce_eager=test_info.enforce_eager,
        max_model_len=test_info.max_model_len,
        max_num_seqs=test_info.max_num_seqs,
        hf_output_post_proc=test_info.hf_output_post_proc,
        vllm_output_post_proc=test_info.vllm_output_post_proc,
        auto_cls=test_info.auto_cls,
        use_tokenizer_eos=test_info.use_tokenizer_eos,
        postprocess_inputs=test_info.postprocess_inputs,
        comparator=test_info.comparator,
        get_stop_token_ids=test_info.get_stop_token_ids,
        limit_mm_per_prompt={"image": 2},
        model_kwargs=test_info.model_kwargs,
    )


### Core test implementation & details
# run_test() is does the heavy lifting of every test type here.
def run_test(
    *,
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    inputs: List[Tuple[List[str], List[Union[List[Image], Image]]]],
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    enforce_eager: bool,
    max_model_len: int,
    max_num_seqs: int,
    hf_output_post_proc: Optional[Callable],
    vllm_output_post_proc: Optional[Callable],
    auto_cls: Type[_BaseAutoModelClass],
    use_tokenizer_eos: bool,
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding],
    comparator: Callable,
    get_stop_token_ids: Optional[Callable],
    limit_mm_per_prompt: Dict[str, int],
    vllm_embeddings: Optional[torch.Tensor]=None,
    model_kwargs: Optional[Dict[str, Any]]=None,
):
    # In the case of embeddings, vLLM takes separate input tensors
    vllm_inputs = vllm_embeddings if vllm_embeddings is not None else inputs
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).
    opt_vllm_kwargs = {}
    if get_stop_token_ids is not None:
        opt_vllm_kwargs["stop_token_ids"] = get_stop_token_ids(tokenizer)

    with vllm_runner(model,
                     max_model_len=max_model_len,
                     max_num_seqs=max_num_seqs,
                     dtype=dtype,
                     limit_mm_per_prompt=limit_mm_per_prompt,
                     tensor_parallel_size=tensor_parallel_size,
                     enforce_eager=enforce_eager) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images,
                                                **opt_vllm_kwargs)
            for prompts, images in vllm_inputs
        ]

    # Some models need to explicitly pass the eos_token_id off the tokenizer or
    # processor for a good comparison; currently assume processor/tokenizer
    # agree on the EOS, and pull it off the tokenizer if requested.
    opt_hf_kwargs = {}
    if use_tokenizer_eos:
        opt_hf_kwargs["eos_token_id"] = tokenizer.eos_token_id

    hf_model = hf_runner(model, dtype=dtype, auto_cls=auto_cls,
                         postprocess_inputs=postprocess_inputs,
                         model_kwargs=model_kwargs)
    with hf_model, torch.no_grad():
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
                                                    tokenizer=tokenizer,
                                                    **opt_hf_kwargs)
            for prompts, images in inputs
        ]

    # Apply output processing / sanitation to the vLLM and HF runner results
    if vllm_output_post_proc is not None:
        vllm_outputs_per_image = [[
            vllm_output_post_proc(res, model) for res in vllm_outputs
        ] for vllm_outputs in vllm_outputs_per_image]

    if hf_output_post_proc is not None:
        hf_outputs_per_image = [[
            hf_output_post_proc(res, model) for res in hf_outputs
        ] for hf_outputs in hf_outputs_per_image]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_image,
                                        vllm_outputs_per_image):
        # This is usually check_logprobs_close, but it's passed through to
        # allow things like check_outputs_equal where needed
        comparator(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )