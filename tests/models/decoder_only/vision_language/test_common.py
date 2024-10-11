"""Common tests for testing .generate() functionality for single / multiple
image support for different VLMs in vLLM.
"""
import itertools
import re
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Type, Union)

import pytest
from PIL.Image import Image
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForVision2Seq, AutoTokenizer, BatchEncoding)
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from vllm.multimodal.utils import rescale_image_size
from vllm.sequence import SampleLogprobs
from vllm.utils import identity, is_cpu, STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from ...utils import check_logprobs_close, check_outputs_equal


### Test Info / Common Configuration
class VLMTestInfo(NamedTuple):
    models: Union[Tuple[str], str]
    prompt_formatter: Callable
    supports_multi_image: bool = False
    img_idx_to_prompt: Callable = lambda idx: "<image>\n"
    # Exposed options for vLLM runner; we change these in a several tests,
    # but the defaults are derived from VllmRunner & the engine defaults
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    max_model_len: int = 1024
    max_num_seqs: int = 256
    use_tokenizer_eos: bool = False
    # Exposed options for HF runner
    model_kwargs: Optional[Dict[str, Any]] = None
    auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM
    # Callable to pass to the HF runner to run on inputs 
    postprocess_inputs: Callable[[BatchEncoding], BatchEncoding] = identity

    # Post processors that if defined, will run oun the outputs of the
    # vLLM and HF runner, respectively (useful for sanitization, etc).
    vllm_output_post_proc: Optional[Callable] = None
    hf_output_post_proc: Optional[Callable] = None
    comparator: Callable = check_logprobs_close
    # Default expandable params per test; these defaults can be overridden in
    # instances of this object; the complete set of test cases for the model
    # is all combinations of .models + all fields below
    max_tokens: Union[int, Tuple[int]] = 128
    num_logprobs: Union[int, Tuple[int]] = 5
    dtype: Union[str] = "half"
    # No image, single-scale, batched single-scale, batched multi-scale
    # size_factors: Tuple[float] = ((1.0,),)
    size_factors: Tuple[float] = ((), (1.0,), (1.0, 1.0, 1.0), (0.25, 0.5, 1.0))


### Base prompts / Common formatting utils
TEST_IMG_PLACEHOLDER = "<image>"
SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    f"{TEST_IMG_PLACEHOLDER}What's the content of the image?",
    "cherry_blossom":
    f"{TEST_IMG_PLACEHOLDER}What is the season?",
})

MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"  # noqa: E501


def replace_test_img_placeholders(prompt: str,
                                  img_idx_to_prompt: Callable) -> str:
    """Given a prompt, replaces each TEST_IMG_PLACEHOLDER with the
    model-specific image prompt.
    """
    prompt_segments = prompt.split(TEST_IMG_PLACEHOLDER)
    img_prompt = prompt_segments[0]
    for placeholder_idx, next_seg in enumerate(prompt_segments[1:], start=1):
        img_prompt += img_idx_to_prompt(placeholder_idx)
        img_prompt += next_seg
    return img_prompt


def get_model_prompts(base_prompts: Union[List[str], Tuple[str]],
                      img_idx_to_prompt: Callable,
                      prompt_formatter: Callable) -> List[str]:
    """Given a model-agnostic base prompt and test configuration for a model(s)
    to be tested, update the image placeholders and apply the prompt formatting
    to get the test prompt string for this model.

    Example for phi3v, given the base_prompt: "<image>What is the season?"
        1. Replace img placeholder(s)
          -> "<|image_1|>\nWhat is the season?"
        2. Apply prompt formatter:
          -> <|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n
    """
    assert isinstance(base_prompts, (list, tuple))
    model_prompts = []
    for base_prompt in base_prompts:
        # Replace the image placeholders in the base prompt with
        # the correct ones for the model that we are testing
        base_prompt_with_imgs = replace_test_img_placeholders(
            base_prompt, img_idx_to_prompt)
        # Apply the prompt formatter to wrap the base prompt with
        # the correct img placeholders to get the model test prompt
        model_prompt = prompt_formatter(base_prompt_with_imgs)
        model_prompts.append(model_prompt)
    return model_prompts


def get_parametrized_options(test_settings: Dict[str, VLMTestInfo],
                             is_multi_image: bool):
    """Converts all of our VLMTestInfo into an expanded list of parameters."""
    if is_multi_image:
        test_settings = {
            model_type: test_info
            for model_type, test_info in test_settings.items()
            if test_info.supports_multi_image
        }

    # Ensure that something is wrapped as an iterable it's not already
    ensure_wrapped = lambda e: e if isinstance(e, (list, tuple)) else (e, )

    def get_model_type_cases(model_type: str, test_info: VLMTestInfo):
        # This is essentially the same as nesting a bunch of mark.parametrize
        # decorators, but we do it programmatically to allow overrides for on
        # a per-model basis, while still being able to execute each of these
        # as individual test cases in pytest.
        return list(
            itertools.product(
                ensure_wrapped(model_type),
                ensure_wrapped(test_info.models),
                ensure_wrapped(test_info.max_tokens),
                ensure_wrapped(test_info.num_logprobs),
                ensure_wrapped(test_info.dtype),
                ensure_wrapped(test_info.size_factors),
            ))

    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    cases_by_model_type = [
        get_model_type_cases(model_type, test_info)
        for model_type, test_info in test_settings.items()
    ]
    return list(itertools.chain(*cases_by_model_type))


### Sanitation utilities for different models
def blip2_vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                               Optional[SampleLogprobs]],
                            model: str):
    """Sanitize vllm output [blip2 models] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(hf_output_str)
    assert hf_output_ids[0] == tokenizer.bos_token_id
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs


def fuyu_vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                              Optional[SampleLogprobs]],
                           model: str):
    """Sanitize vllm output [fuyu models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    hf_output_str = output_str.lstrip() + "|ENDOFTEXT|"

    return output_ids, hf_output_str, out_logprobs


def llava_vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                               Optional[SampleLogprobs]],
                            model: str):
    """Sanitize vllm output [Llava models] to be comparable with hf output."""
    output_ids, output_str, out_logprobs = vllm_output

    config = AutoConfig.from_pretrained(model)
    image_token_id = config.image_token_index

    tokenizer = AutoTokenizer.from_pretrained(model)
    eos_token_id = tokenizer.eos_token_id

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]

    assert output_str[0] == " "
    hf_output_str = output_str[1:]
    if hf_output_ids[-1] == eos_token_id:
        hf_output_str = hf_output_str + tokenizer.decode(eos_token_id)

    return hf_output_ids, hf_output_str, out_logprobs


def phi3v_vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                               Optional[SampleLogprobs]],
                            model: str):
    """Sanitize vllm output [phi3v] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    output_str_without_image = re.sub(r"(<\|image_\d+\|>)+", "", output_str)
    assert output_str_without_image[0] == " "
    output_str_without_image = output_str_without_image[1:]

    hf_output_str = output_str_without_image + "<|end|><|endoftext|>"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    assert hf_output_ids[0] == 1
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs

### Common postprocessors to run on HF BatchEncoding
# NOTE: It would be helpful to rewrite this to be configured in the test info,
# but built inside of the test so that we don't need to specify the dtype twice
def get_key_type_post_processor(
        hf_inp_key: str,
        dtype: str
    ) -> Callable[[BatchEncoding], BatchEncoding]:
    """Gets a handle to a post processor which converts a """
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
    def process(hf_inputs: BatchEncoding):
        hf_inputs[hf_inp_key] = hf_inputs[hf_inp_key].to(torch_dtype)
        return hf_inputs
    return process

###
# NOTE: Convention here is to map the names of the file containing multimodal
# models in alphabetical order to test info so that we can easily tell which
# models have tests.
# yapf: disable
VLM_TEST_SETTINGS = {
    "blip2": VLMTestInfo(
        models="Salesforce/blip2-opt-2.7b",
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        vllm_output_post_proc=blip2_vllm_to_hf_output,
        auto_cls=AutoModelForVision2Seq,
    ),
    "fuyu": VLMTestInfo(
        models="adept/fuyu-8b",
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        vllm_output_post_proc=fuyu_vllm_to_hf_output,
        dtype="bfloat16" if is_cpu() else "half",
        num_logprobs=10,
        max_model_len=2048,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        size_factors=((), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)),
    ),
    "chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        dtype="bfloat16",
        postprocess_inputs=get_key_type_post_processor(
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

    ## Tests beyond this point have been validated to align with current tests 
    # "intern_vl": VLMTestInfo(
    #     models=["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B"],
    #     supports_multi_image=True,
    #     prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
    #     dtype="bfloat16" if is_cpu() else "half",
    #     num_logprobs=10,
    #     max_model_len=4096,
    # ),
    # "llava": VLMTestInfo(
    #     models="llava-hf/llava-1.5-7b-hf",
    #     prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
    #     vllm_output_post_proc=llava_vllm_to_hf_output,
    # ),
    # "minicpmv": VLMTestInfo(
    #     models="openbmb/MiniCPM-Llama3-V-2_5",
    #     supports_multi_image=True,
    #     img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
    #     prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
    # ),
    # "phi3v": VLMTestInfo(
    #     models="microsoft/Phi-3.5-vision-instruct",
    #     supports_multi_image=True,
    #     img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
    #     prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n", # noqa: E501
    #     vllm_output_post_proc=phi3v_vllm_to_hf_output,
    #     num_logprobs=10,
    # ),
    # "qwen": VLMTestInfo(
    #     models="Qwen/Qwen-VL",
    #     supports_multi_image=True,
    #     img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
    #     prompt_formatter=lambda img_prompt: f"{img_prompt} ",
    # ),
}
# yapf: enable

@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_factors",
    get_parametrized_options(VLM_TEST_SETTINGS, is_multi_image=False))
def test_single_image_generation(model_type: str, model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 size_factors: Tuple[float], hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompts = get_model_prompts(SINGLE_IMAGE_BASE_PROMPTS,
                                      test_info.img_idx_to_prompt,
                                      test_info.prompt_formatter)

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
    )


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
):
    # NOTE: take care of the order. run vLLM first, and then run HF.
    # vLLM needs a fresh new process without cuda initialization.
    # if we run HF first, the cuda initialization will be done and it
    # will hurt multiprocessing backend with fork method (the default method).

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model,
                     max_model_len=max_model_len,
                     max_num_seqs=max_num_seqs,
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     enforce_eager=enforce_eager) as vllm_model:
        vllm_outputs_per_image = [
            vllm_model.generate_greedy_logprobs(prompts,
                                                max_tokens,
                                                num_logprobs=num_logprobs,
                                                images=images)
            for prompts, images in inputs
        ]

    # Some models need to explicitly pass the eos_token_id off the tokenizer or
    # processor for a good comparison; currently assume processor/tokenizer
    # agree on the EOS, and pull it off the tokenizer if requested.
    opt_hf_kwargs = {}
    if use_tokenizer_eos:
        opt_hf_kwargs["eos_token_id"] = AutoTokenizer.from_pretrained(
            model
        ).eos_token_id

    with hf_runner(model, dtype=dtype, auto_cls=auto_cls, postprocess_inputs=postprocess_inputs) as hf_model:
        hf_outputs_per_image = [
            hf_model.generate_greedy_logprobs_limit(prompts,
                                                    max_tokens,
                                                    num_logprobs=num_logprobs,
                                                    images=images,
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
