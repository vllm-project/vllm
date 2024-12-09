from array import array
from typing import Callable, Dict, Mapping, Optional
from unittest.mock import patch

import pytest
import torch

from vllm.inputs import (DecoderOnlyInputs, DummyData, InputContext,
                         InputRegistry, ProcessorInputs, token_inputs)
from vllm.multimodal import MultiModalRegistry
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData

from ..models.utils import build_model_context

# Used for fast tests where the model doesn't matter
DUMMY_MODEL_ID = "facebook/opt-125m"
# Used for tests that need a multimodal model
MULTIMODAL_MODEL_ID = "OpenGVLab/InternVL2-2B"

# For mm_processor_kwargs - we test overrides by defining mocks for each place
# it is used, and ensuring that we can pass processor kwargs an override value
# to receive the intended result for things like sequence length etc.
DEFAULT_MAX_DYNAMIC_PATCH = 6
MAX_DYNAMIC_PATCH_OVERRIDE = 4


# Mocks for all of the places that we use the mm_processor_kwargs
# to override values in different callables
@pytest.fixture
def use_processor_mock():
    """Patches the internal model input processor with an override callable."""

    def custom_processor(ctx: InputContext,
                         inputs: DecoderOnlyInputs,
                         *,
                         max_dynamic_patch=DEFAULT_MAX_DYNAMIC_PATCH):
        # For testing purposes, we don't worry about the prompt
        return token_inputs(
            prompt_token_ids=[],
            mm_processor_kwargs={"max_dynamic_patch": max_dynamic_patch})

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
                                  max_dynamic_patch=DEFAULT_MAX_DYNAMIC_PATCH):
        seq_data = SequenceData(
            array(VLLM_TOKEN_ID_ARRAY_TYPE, [0] * max_dynamic_patch))
        return DummyData(seq_data, None)

    with patch(
            "vllm.inputs.registry.InputRegistry._default_dummy_data_factory",
            custom_dummy_data_factory):
        yield


# Lazy import to avoid CUDA reinitialization error
def mm_model_cls():
    from vllm.model_executor.models.internvl import InternVLChatModel

    return InternVLChatModel


# lambda whose signature matches max token calcs extra & mapper + extra kwargs
get_max_dynamic_patch = lambda ctx, *, max_dynamic_patch=DEFAULT_MAX_DYNAMIC_PATCH: max_dynamic_patch  # noqa: E501
custom_mapper = lambda ctx, data, *, max_dynamic_patch=DEFAULT_MAX_DYNAMIC_PATCH: {  # noqa: E501
    "pixel_values": torch.zeros(size=(1, max_dynamic_patch + 1, 3, 448, 448))
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


def _get_max_dynamic_patch_info(init_max_dynamic_patch: int,
                                inference_max_dynamic_patch: int):
    """Get the init / inference kwargs and expected max_dynamic_patch."""
    # If we have a value for max_dynamic_patch, pass the override value and make
    # sure we get that value as a return-value from out mock processor,
    # otherwise fall back to the default value
    init_kwargs = None if init_max_dynamic_patch is None else {
        "max_dynamic_patch": init_max_dynamic_patch
    }
    inference_kwargs = None if inference_max_dynamic_patch is None else {
        "max_dynamic_patch": inference_max_dynamic_patch
    }
    if inference_max_dynamic_patch is not None:
        expected_seq_count = inference_max_dynamic_patch
    elif init_max_dynamic_patch is not None:
        expected_seq_count = init_max_dynamic_patch
    else:
        expected_seq_count = DEFAULT_MAX_DYNAMIC_PATCH
    return init_kwargs, inference_kwargs, expected_seq_count


def _get_processed_max_dynamic_patch(
    processor: Callable[[ProcessorInputs], ProcessorInputs],
    inference_kwargs: Optional[Dict[str, int]],
) -> int:
    processed_inputs = processor(
        token_inputs(prompt_token_ids=[],
                     prompt="",
                     mm_processor_kwargs=inference_kwargs))

    assert "type" in processed_inputs
    assert processed_inputs["type"] == "token"
    assert "mm_processor_kwargs" in processed_inputs
    return processed_inputs["mm_processor_kwargs"]["max_dynamic_patch"]


@pytest.mark.parametrize(
    "init_max_dynamic_patch,inference_max_dynamic_patch", [
        (None, None),
        (MAX_DYNAMIC_PATCH_OVERRIDE, None),
        (DEFAULT_MAX_DYNAMIC_PATCH, MAX_DYNAMIC_PATCH_OVERRIDE),
    ])
def test_input_processor_kwargs(use_processor_mock, init_max_dynamic_patch,
                                inference_max_dynamic_patch):
    """Ensure input processors can use processor kwargs."""
    dummy_registry = InputRegistry()

    (init_kwargs, inference_kwargs,
     expected_seq_count) = _get_max_dynamic_patch_info(
         init_max_dynamic_patch, inference_max_dynamic_patch)

    ctx = build_model_context(DUMMY_MODEL_ID, mm_processor_kwargs=init_kwargs)
    processor = dummy_registry.create_input_processor(ctx.model_config)
    max_dynamic_patch_val = _get_processed_max_dynamic_patch(
        processor, inference_kwargs)

    assert max_dynamic_patch_val == expected_seq_count


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
    max_dynamic_patch_val = _get_processed_max_dynamic_patch(
        processor, mm_processor_kwargs)
    assert max_dynamic_patch_val == DEFAULT_MAX_DYNAMIC_PATCH


### Test overrides for the dummy data
@pytest.mark.parametrize("max_dynamic_patch",
                         [None, MAX_DYNAMIC_PATCH_OVERRIDE])
def test_dummy_data_kwarg_overrides(use_dummy_data_mock, max_dynamic_patch):
    """Ensure dummy data factories can use processor kwargs."""
    mm_processor_kwargs = None if max_dynamic_patch is None else {
        "max_dynamic_patch": max_dynamic_patch
    }
    expected_seq_count = (DEFAULT_MAX_DYNAMIC_PATCH
                          if max_dynamic_patch is None else max_dynamic_patch)
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
    assert len(
        dummy_data.seq_data.prompt_token_ids) == DEFAULT_MAX_DYNAMIC_PATCH


### Test overrides for the max token count per multimodal instance
@pytest.mark.parametrize("max_dynamic_patch",
                         [None, MAX_DYNAMIC_PATCH_OVERRIDE])
def test_max_tokens_kwarg_overrides(max_dynamic_patch):
    """Ensure max token calcs can use processor kwargs."""
    mm_processor_kwargs = None if max_dynamic_patch is None else {
        "max_dynamic_patch": max_dynamic_patch
    }
    expected_seq_count = (DEFAULT_MAX_DYNAMIC_PATCH
                          if max_dynamic_patch is None else max_dynamic_patch)

    ctx = build_model_context(MULTIMODAL_MODEL_ID,
                              task="generate",
                              trust_remote_code=True,
                              mm_processor_kwargs=mm_processor_kwargs,
                              limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)
    # Patch the image registry for phi3v with our lambda that is compatible
    # with overrides, then ensure that calling the method correctly echos
    # our max_dynamic_patch value back from the mm_processor_kwargs.
    with patch.object(
            mm_registry._get_plugin("image"),
            "_max_mm_tokens",
        {mm_model_cls(): get_max_dynamic_patch},
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
        {mm_model_cls(): get_max_dynamic_patch},
    ):
        max_multimodal_tokens = mm_registry.get_max_multimodal_tokens(
            ctx.model_config)

    assert max_multimodal_tokens == DEFAULT_MAX_DYNAMIC_PATCH


### Test overrides for the mapper
@pytest.mark.parametrize(
    "max_dynamic_patch",
    [DEFAULT_MAX_DYNAMIC_PATCH, MAX_DYNAMIC_PATCH_OVERRIDE])
def test_default_mapper_with_processor_kwargs(image_assets, max_dynamic_patch):
    """Ensure that the mapper processor kwargs can fall back to HF models."""
    # NOTE - we don't validate bad inputs for the default mapper, because it's
    # through the automodel interface in transformers, so we can't easily
    # inspect what kwargs are or are not allowed.
    ctx = build_model_context(
        MULTIMODAL_MODEL_ID,
        task="generate",
        trust_remote_code=True,
        mm_processor_kwargs={"max_dynamic_patch": max_dynamic_patch},
        limit_mm_per_prompt={"image": 1})

    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(ctx.model_config)

    image = image_assets[0].pil_image
    mm_inputs = {"image": image}

    mapped_inputs = mm_registry.map_input(ctx.model_config, mm_inputs)
    # pixel vals should have shape: [batch, max_dynamic_patch+1, ...]
    assert mapped_inputs["pixel_values"].shape[1] == max_dynamic_patch + 1


@pytest.mark.parametrize(
    "init_max_dynamic_patch,inference_max_dynamic_patch", [
        (None, None),
        (MAX_DYNAMIC_PATCH_OVERRIDE, None),
        (DEFAULT_MAX_DYNAMIC_PATCH, MAX_DYNAMIC_PATCH_OVERRIDE),
    ])
def test_custom_mapper_kwarg_overrides(image_assets, init_max_dynamic_patch,
                                       inference_max_dynamic_patch):
    """Ensure custom mappers can use processor kwargs."""
    (init_kwargs, inference_kwargs,
     expected_seq_count) = _get_max_dynamic_patch_info(
         init_max_dynamic_patch, inference_max_dynamic_patch)

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
    # our max_dynamic_patch value back from the mm_processor_kwargs.
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
    # our max_dynamic_patch value back from the mm_processor_kwargs.
    mm_registry._get_plugin("image").register_input_mapper(custom_mapper)(
        mm_model_cls())
    # Should filter out the inference time kwargs
    mapped_inputs = mm_registry.map_input(
        ctx.model_config, mm_inputs, mm_processor_kwargs=mm_processor_kwargs)

    assert mapped_inputs["pixel_values"].shape[1] == (
        DEFAULT_MAX_DYNAMIC_PATCH + 1)
