from array import array
from typing import Mapping
from unittest.mock import patch

import pytest
import torch

from vllm.inputs import (DecoderOnlyInputs, DummyData, InputContext,
                         InputRegistry, token_inputs)
from vllm.multimodal import MultiModalRegistry
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData

from ..models.utils import build_model_context

# Used for fast tests where the model doesn't matter
DUMMY_MODEL_ID = "facebook/opt-125m"
# Used for tests that need a multimodal model
MULTIMODAL_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

# For mm_processor_kwargs - we test overrides by defining mocks for each place
# it is used, and ensuring that we can pass processor kwargs an override value
# to receive the intended result for things like sequence length etc.
DEFAULT_NUM_CROPS = 4
NUM_CROPS_OVERRIDE = 16


# Mocks for all of the places that we use the mm_processor_kwargs
# to override values in different callables
@pytest.fixture
def use_processor_mock():
    """Patches the internal model input processor with an override callable."""

    def custom_processor(ctx: InputContext,
                         inputs: DecoderOnlyInputs,
                         *,
                         num_crops=DEFAULT_NUM_CROPS):
        # For testing purposes, we don't worry about the llm inputs / return
        # type validation, and just return the value of the kwarg that we
        # clobber.
        return num_crops

    with patch("vllm.inputs.registry.InputRegistry._get_model_input_processor",
               return_value=custom_processor):
        yield


@pytest.fixture
def use_dummy_data_mock():
    """Patches the internal model input processor with an override callable."""

    def custom_dummy_data_factory(self,
                                  ctx: InputContext,
                                  seq_len: int,
                                  mm_counts: Mapping[str, int],
                                  *,
                                  num_crops=DEFAULT_NUM_CROPS):
        seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE, [0] * num_crops))
        return DummyData(seq_data, None)

    with patch(
            "vllm.inputs.registry.InputRegistry._default_dummy_data_factory",
            custom_dummy_data_factory):
        yield


# Lazy import to avoid CUDA reinitialization error
def mm_model_cls():
    from vllm.model_executor.models.phi3v import Phi3VForCausalLM

    return Phi3VForCausalLM


# lambda whose signature matches max token calcs extra & mapper + extra kwargs
get_num_crops = lambda ctx, *, num_crops=DEFAULT_NUM_CROPS: num_crops
custom_mapper = lambda ctx, data, *, num_crops=DEFAULT_NUM_CROPS: {
    "pixel_values": torch.zeros(size=(1, num_crops + 1, 3, 336, 336))
}


### Tests for default processor logic & mm_processor_kwargs wrapping
def test_default_processor_is_a_noop():
    """Ensure that by default, there is no processor override."""
    dummy_registry = InputRegistry()
    ctx = build_model_context(DUMMY_MODEL_ID)
    processor = dummy_registry.create_input_processor(ctx.model_config)
    proc_inputs = token_inputs(prompt_token_ids=[], prompt="")
    proc_outputs = processor(inputs=proc_inputs)
    assert proc_inputs is proc_outputs


def _get_num_crops_info(init_num_crops: int, inference_num_crops: int):
    """Get the init / inference kwargs and expected num_crops for this test."""
    # If we have a value for num_crops, pass the override value and make
    # sure we get that value as a return-value from out mock processor,
    # otherwise fall back to the default value
    init_kwargs = None if init_num_crops is None else {
        "num_crops": init_num_crops
    }
    inference_kwargs = None if inference_num_crops is None else {
        "num_crops": inference_num_crops
    }
    if inference_num_crops is not None:
        expected_seq_count = inference_num_crops
    elif init_num_crops is not None:
        expected_seq_count = init_num_crops
    else:
        expected_seq_count = DEFAULT_NUM_CROPS
    return init_kwargs, inference_kwargs, expected_seq_count


@pytest.mark.parametrize("init_num_crops,inference_num_crops", [
    (None, None),
    (NUM_CROPS_OVERRIDE, None),
    (DEFAULT_NUM_CROPS, NUM_CROPS_OVERRIDE),
])
def test_input_processor_kwargs(use_processor_mock, init_num_crops,
                                inference_num_crops):
    """Ensure input processors can use processor kwargs."""
    dummy_registry = InputRegistry()

    init_kwargs, inference_kwargs, expected_seq_count = _get_num_crops_info(
        init_num_crops, inference_num_crops)

    ctx = build_model_context(DUMMY_MODEL_ID, mm_processor_kwargs=init_kwargs)
    processor = dummy_registry.create_input_processor(ctx.model_config)
    num_crops_val = processor(
        token_inputs(prompt_token_ids=[],
                     prompt="",
                     mm_processor_kwargs=inference_kwargs))
    assert num_crops_val == expected_seq_count


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        # Not part of the signature
        {
            "does_not_exist": 100
        },
        # Part of the signature, not keyword only
        {
            "ctx": "something bad"
        }
    ])
def test_processor_with_sad_kwarg_overrides(use_processor_mock,
                                            mm_processor_kwargs):
    """Ensure that input processors filter out invalid mm_processor_kwargs"""
    dummy_registry = InputRegistry()
    # Should filter out the init time kwargs
    ctx = build_model_context(DUMMY_MODEL_ID,
                              mm_processor_kwargs=mm_processor_kwargs)

    processor = dummy_registry.create_input_processor(ctx.model_config)
    # Should filter out the inference time kwargs
    num_crops_val = processor(
        token_inputs(prompt_token_ids=[],
                     prompt="",
                     mm_processor_kwargs=mm_processor_kwargs))
    assert num_crops_val == DEFAULT_NUM_CROPS


### Test overrides for the dummy data
@pytest.mark.parametrize("num_crops", [None, NUM_CROPS_OVERRIDE])
def test_dummy_data_kwarg_overrides(use_dummy_data_mock, num_crops):
    """Ensure dummy data factories can use processor kwargs."""
    mm_processor_kwargs = None if num_crops is None else {
        "num_crops": num_crops
    }
    expected_seq_count = DEFAULT_NUM_CROPS if num_crops is None else num_crops
    dummy_registry = InputRegistry()
    ctx = build_model_context(DUMMY_MODEL_ID,
                              mm_processor_kwargs=mm_processor_kwargs)
    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    # NOTE: seq_len is thrown away here since this will leverage the
    # default dummy data factory that we have patched in, whose seq
    # len is solely dependent on the value of the mm_processor_kwargs.
    dummy_data = dummy_registry.dummy_data_for_profiling(
        ctx.model_config, seq_len=-1, mm_registry=mm_registry)
    assert len(dummy_data.seq_data.prompt_token_ids) == expected_seq_count


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        # Not part of the signature
        {
            "does_not_exist": 100
        },
        # Part of the signature, not keyword only
        {
            "ctx": "something bad"
        }
    ])
def test_dummy_data_with_sad_kwarg_overrides(use_dummy_data_mock,
                                             mm_processor_kwargs):
    """Ensure the dummy data factory filters out invalid mm_processor_kwargs"""
    dummy_registry = InputRegistry()
    ctx = build_model_context(DUMMY_MODEL_ID,
                              mm_processor_kwargs=mm_processor_kwargs)
    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    # NOTE: seq_len is thrown away here since this will leverage the
    # default dummy data factory that we have patched in, whose seq
    # len is solely dependent on the value of the mm_processor_kwargs.
    dummy_data = dummy_registry.dummy_data_for_profiling(
        ctx.model_config, seq_len=-1, mm_registry=mm_registry)
    assert len(dummy_data.seq_data.prompt_token_ids) == DEFAULT_NUM_CROPS


### Test overrides for the max token count per multimodal instance
@pytest.mark.parametrize("num_crops", [None, NUM_CROPS_OVERRIDE])
def test_max_tokens_kwarg_overrides(num_crops):
    """Ensure max token calcs can use processor kwargs."""
    mm_processor_kwargs = None if num_crops is None else {
        "num_crops": num_crops
    }
    expected_seq_count = DEFAULT_NUM_CROPS if num_crops is None else num_crops

    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs=mm_processor_kwargs,
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)
    # Patch the image registry for phi3v with our lambda that is compatible
    # with overrides, then ensure that calling the method correctly echos
    # our num_crops value back from the mm_processor_kwargs.
    with patch.object(
            mm_registry._get_plugin("image"),
            "_max_mm_tokens",
        {mm_model_cls(): get_num_crops},
    ):
        max_multimodal_tokens = mm_registry.get_max_multimodal_tokens(
            ctx.model_config)

    assert expected_seq_count == max_multimodal_tokens


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        # Not part of the signature
        {
            "does_not_exist": 100
        },
        # Part of the signature, not keyword only
        {
            "ctx": "something bad"
        }
    ])
def test_max_tokens_with_sad_kwarg_overrides(mm_processor_kwargs):
    """Ensure that max token calcs filters out invalid mm_processor_kwargs"""
    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs=mm_processor_kwargs,
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    # Similar before, but since these kwargs get filtered,
    # we always get our default value back.
    with patch.object(
            mm_registry._get_plugin("image"),
            "_max_mm_tokens",
        {mm_model_cls(): get_num_crops},
    ):
        max_multimodal_tokens = mm_registry.get_max_multimodal_tokens(
            ctx.model_config)

    assert max_multimodal_tokens == DEFAULT_NUM_CROPS


### Test overrides for the mapper
@pytest.mark.parametrize("num_crops", [DEFAULT_NUM_CROPS, NUM_CROPS_OVERRIDE])
def test_default_mapper_with_processor_kwargs(image_assets, num_crops):
    """Ensure that the mapper processor kwargs can fall back to HF models."""
    # NOTE - we don't validate bad inputs for the default mapper, because it's
    # through the automodel interface in transformers, so we can't easily
    # inspect what kwargs are or are not allowed.
    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs={"num_crops": num_crops},
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    image = image_assets[0].pil_image
    mm_inputs = {"image": image}

    mapped_inputs = mm_registry.map_input(ctx.model_config, mm_inputs)
    # Phi3v pixel vals should have shape: [batch, num_crops+1, 3, 336, 336]
    assert mapped_inputs["pixel_values"].shape[1] == num_crops + 1


@pytest.mark.parametrize("init_num_crops,inference_num_crops", [
    (None, None),
    (NUM_CROPS_OVERRIDE, None),
    (DEFAULT_NUM_CROPS, NUM_CROPS_OVERRIDE),
])
def test_custom_mapper_kwarg_overrides(image_assets, init_num_crops,
                                       inference_num_crops):
    """Ensure custom mappers can use processor kwargs."""
    init_kwargs, inference_kwargs, expected_seq_count = _get_num_crops_info(
        init_num_crops, inference_num_crops)

    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs=init_kwargs,
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)
    image = image_assets[0].pil_image
    mm_inputs = {"image": image}

    # Patch the image registry for phi3v with our lambda that is compatible
    # with overrides, then ensure that calling the method correctly echos
    # our num_crops value back from the mm_processor_kwargs.
    mm_registry._get_plugin("image").register_input_mapper(custom_mapper)(
        mm_model_cls())
    mapped_inputs = mm_registry.map_input(ctx.model_config, mm_inputs,
                                          inference_kwargs)

    assert mapped_inputs["pixel_values"].shape[1] == expected_seq_count + 1


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [
        # Not part of the signature
        {
            "does_not_exist": 100
        },
        # Part of the signature, not keyword only
        {
            "ctx": "something bad"
        }
    ])
def test_custom_mapper_with_sad_kwarg_overrides(image_assets,
                                                mm_processor_kwargs):
    """Ensure that custom mappers filters out invalid mm_processor_kwargs"""
    # Should filter out the init time kwargs
    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs=mm_processor_kwargs,
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)
    image = image_assets[0].pil_image
    mm_inputs = {"image": image}

    # Patch the image registry for phi3v with our lambda that is compatible
    # with overrides, then ensure that calling the method correctly echos
    # our num_crops value back from the mm_processor_kwargs.
    mm_registry._get_plugin("image").register_input_mapper(custom_mapper)(
        mm_model_cls())
    # Should filter out the inference time kwargs
    mapped_inputs = mm_registry.map_input(
        ctx.model_config, mm_inputs, mm_processor_kwargs=mm_processor_kwargs)

    assert mapped_inputs["pixel_values"].shape[1] == DEFAULT_NUM_CROPS + 1
