"""Common tests for testing .generate() functionality for single / multiple
image support for different VLMs in vLLM.
"""
import itertools
import re
import pytest
from typing import NamedTuple, Callable, Union, List, Tuple, Optional, Dict
from transformers import AutoTokenizer, AutoConfig
from vllm.sequence import SampleLogprobs
from vllm.utils import is_cpu
from vllm.multimodal.utils import rescale_image_size

from ....conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from ...utils import check_logprobs_close


### Test Info / Common Configuration
class VLMTestInfo(NamedTuple):
    models: Union[Tuple[str], str]
    prompt_formatter: Callable
    supports_multi_image: bool = False
    img_idx_to_prompt: Callable = lambda idx: "<image>\n"
    sanitizer: Optional[Callable] = None
    # Default expandable params per test; these defaults can be overridden in
    # instances of this object; the complete set of test cases for the model
    # is all combinations of .models + all fields below 
    max_tokens: Union[int, Tuple[int]]=128
    num_logprobs: Union[int, Tuple[int]]=5
    target_dtype: Union[Tuple[int]]="half"

### Base prompts / Common formatting utils
TEST_IMG_PLACEHOLDER= "<image>"
SINGLE_IMAGE_BASE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign": f"{TEST_IMG_PLACEHOLDER}What's the content of the image?\n",
    "cherry_blossom": f"{TEST_IMG_PLACEHOLDER}What is the season?\n",
})

MULTI_IMAGE_BASE_PROMPT = f"Image-1: {TEST_IMG_PLACEHOLDER}Image-2: {TEST_IMG_PLACEHOLDER}Describe the two images in detail.\n"
SIZE_FACTORS = (
    # No image
    (),
    # # Single-scale
    (1.0,),
    # # Single-scale, batched
    (1.0, 1.0, 1.0),
    # # Multi-scale
    (0.25, 0.5, 1.0),
)

def replace_test_img_placeholders(prompt: str, img_idx_to_prompt: Callable) -> str:
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
    assert isinstance(base_prompts, (List, Tuple))
    model_prompts = []
    for base_prompt in base_prompts:
        # Replace the image placeholders in the base prompt with
        # the correct ones for the model that we are testing
        base_prompt_with_imgs = replace_test_img_placeholders(
            base_prompt,
            img_idx_to_prompt
        )
        # Apply the prompt formatter to wrap the base prompt with
        # the correct img placeholders to get the model test prompt
        model_prompt = prompt_formatter(base_prompt_with_imgs)
        model_prompts.append(model_prompt)
    return model_prompts


def get_parametrized_options(test_settings: Dict[str, VLMTestInfo], is_multi_image: bool):
    """Converts all of our VLMTestInfo into an expanded list of parameters."""
    if is_multi_image:
        test_settings = {
            model_type: test_info
            for model_type, test_info 
            in test_settings.items() 
            if test_info.supports_multi_image
        }

    # Ensure that something is wrapped as an iterable it's not already
    ensure_wrapped = lambda e: e if isinstance(e, (list, tuple)) else (e,)

    def get_model_type_cases(model_type: str, test_info:VLMTestInfo):
        # This is essentially the same as nesting a bunch of mark.parametrize
        # decorators, but we do it programatically to allow overrides for on
        # a per-model basis, while still being able to execute each of these
        # as individual test cases in pytest.
        return list(itertools.product(
            ensure_wrapped(model_type),
            ensure_wrapped(test_info.models),
            ensure_wrapped(test_info.max_tokens),
            ensure_wrapped(test_info.num_logprobs),
            ensure_wrapped(test_info.target_dtype),
        ))
    # Get a list per model type, where each entry contains a tuple of all of
    # that model type's cases, then flatten them into the top level so that
    # we can consume them in one mark.parametrize call.
    cases_by_model_type = [
        get_model_type_cases(model_type, test_info)
        for model_type, test_info in test_settings.items()]
    return list(itertools.chain(*cases_by_model_type))

### Sanitation utilities for different models
def blip2_vllm_to_hf_output(
        vllm_output: Tuple[List[int], str, Optional[SampleLogprobs]],
        model: str
    ):
    """Sanitize vllm output [blip2 models] to be comparable with hf output."""
    _, output_str, out_logprobs = vllm_output

    hf_output_str = output_str + "\n"

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_output_ids = tokenizer.encode(hf_output_str)
    assert hf_output_ids[0] == tokenizer.bos_token_id
    hf_output_ids = hf_output_ids[1:]

    return hf_output_ids, hf_output_str, out_logprobs

def fuyu_vllm_to_hf_output(vllm_output: Tuple[List[int], str,
                                         Optional[SampleLogprobs]]):
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

### 
# NOTE: Convention here is to map the names of the file containing multimodal
# models in aphabetical order to test info so that we can easily tell which
# models have tests.
VLM_TEST_SETTINGS = {
    "blip2": VLMTestInfo(
        models=["Salesforce/blip2-opt-2.7b"],
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        sanitizer=blip2_vllm_to_hf_output,
    ),
    "chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        target_dtype="bfloat16",
    ),
    "fuyu": VLMTestInfo(
        models="adept/fuyu-8b",
        img_idx_to_prompt=lambda idx: "",
        prompt_formatter=lambda img_prompt: img_prompt,
        sanitizer=fuyu_vllm_to_hf_output,
        target_dtype="bfloat16" if is_cpu() else "half",
        num_logprobs=10,
    ),
    "intern_vl": VLMTestInfo(
        models=["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B"],
        supports_multi_image=True,
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n",
        target_dtype="bfloat16" if is_cpu() else "half",
        num_logprobs=10,
    ),
    "llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        sanitizer=llava_vllm_to_hf_output,
    ),
    "minicpmv": VLMTestInfo(
        models="openbmb/MiniCPM-Llama3-V-2_5",
        supports_multi_image=True,
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
    ),
    "phi3v": VLMTestInfo(
        models="microsoft/Phi-3.5-vision-instruct",
        supports_multi_image=True,
        img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n",
        sanitizer=phi3v_vllm_to_hf_output,
        num_logprobs=10,
    ),
    "qwen": VLMTestInfo(
        models="Qwen/Qwen-VL",
        supports_multi_image=True,
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        prompt_formatter=lambda img_prompt: f"{img_prompt} ",
    ),
}


@pytest.mark.parametrize("size_factors", SIZE_FACTORS)
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,target_dtype",
    get_parametrized_options(VLM_TEST_SETTINGS, is_multi_image=False)
)
def test_single_image_generation(model_type: str,
                                 model: str,
                                 max_tokens: int,
                                 num_logprobs: str,
                                 target_dtype: str,
                                 size_factors: List[float],
                                 hf_runner,
                                 vllm_runner,
                                 image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompts = get_model_prompts(
        SINGLE_IMAGE_BASE_PROMPTS,
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter
    )

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
    # TODO - port the actual test


### Multi-image generation tests [only for VLMs that support it]

@pytest.mark.parametrize("size_factors", SIZE_FACTORS)
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,target_dtype",
    get_parametrized_options(VLM_TEST_SETTINGS, is_multi_image=True)
)
def test_multi_image_generation(model_type: str,
                                 model: str,
                                 max_tokens: int,
                                 num_logprobs: str,
                                 target_dtype: str,
                                 size_factors: List[float],
                                 hf_runner,
                                 vllm_runner,
                                 image_assets: _ImageAssets):
    test_info = VLM_TEST_SETTINGS[model_type]
    model_prompt = get_model_prompts(
        [MULTI_IMAGE_BASE_PROMPT],
        test_info.img_idx_to_prompt,
        test_info.prompt_formatter
    )[0]

    images = [asset.pil_image for asset in image_assets]

    # This is similar to the single image case, but we rescale each of the
    # images in the multi-image prompt; currently we only have one model prompt
    inputs = [
        ([model_prompt for _ in size_factors],
         [[rescale_image_size(image, factor) for image in images]
          for factor in size_factors])
    ]




# """
# TODO  port the actual tests
# TODO  port overrides for vllm runner for stuff like num seqs etc
# TODO  port custom logic stuff that has to happen on the prompt for models like Qwen-VL to run HF
# """
