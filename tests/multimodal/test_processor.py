from unittest.mock import patch

import pytest

from vllm.config import ModelConfig
from vllm.inputs import InputContext, LLMInputs
from vllm.inputs.registry import InputRegistry

DUMMY_MODEL_ID = "facebook/opt-125m"
# For processor kwargs - we test overrides by defining a callable with a
# default for the `num_crops`, then override the value through the processor
# kwargs
DEFAULT_NUM_CROPS = 4
NUM_CROPS_OVERRIDE = 16


@pytest.fixture
def processor_mock():
    """Patches the internal model input processor with an override callable."""

    def custom_processor(ctx: InputContext,
                         llm_inputs: LLMInputs,
                         num_crops=DEFAULT_NUM_CROPS):
        # For testing purposes, we don't worry about the llm inputs / return
        # type validation, and just return the value of the kwarg that we
        # clobber.
        return num_crops

    with patch("vllm.inputs.registry.InputRegistry._get_model_input_processor",
               return_value=custom_processor):
        yield


def get_model_config(processor_kwargs=None):
    """Creates a handle to a model config, which may have processor kwargs."""
    # NOTE - values / architecture don't matter too much here since we patch
    # the return values for stuff like the input processor anyway.
    return ModelConfig(DUMMY_MODEL_ID,
                       DUMMY_MODEL_ID,
                       tokenizer_mode="auto",
                       trust_remote_code=False,
                       dtype="float16",
                       seed=0,
                       processor_kwargs=processor_kwargs)


def test_default_processor_is_a_noop():
    """Ensure that by default, there is no processor override."""
    dummy_registry = InputRegistry()
    model_config = get_model_config()
    processor = dummy_registry.create_input_processor(model_config)
    proc_inputs = LLMInputs(prompt="foobar")
    proc_outputs = processor(inputs=proc_inputs)
    # We should get the same object back since this is a no-op by default
    assert proc_inputs is proc_outputs


def test_processor_default_kwargs(processor_mock):
    """Ensure we can call a processor that has extra kwargs & no overrides."""
    dummy_registry = InputRegistry()
    model_config = get_model_config()
    processor = dummy_registry.create_input_processor(model_config)
    # The patched fixture patches the processor to return the value of
    # num_crops in the processor call, which should be 4 by default.
    num_crops_val = processor(LLMInputs(prompt="foobar"))
    assert num_crops_val == DEFAULT_NUM_CROPS


def test_processor_default_kwargs_with_override(processor_mock):
    """Ensure we can call a processor that has extra kwargs & no overrides."""
    dummy_registry = InputRegistry()
    # Create processor_kwargs to override the value used
    # for num_crops in the patched processor callable
    model_config = get_model_config(
        processor_kwargs={"num_crops": NUM_CROPS_OVERRIDE})
    processor = dummy_registry.create_input_processor(model_config)
    num_crops_val = processor(LLMInputs(prompt="foobar"))
    # Since the patched processor is an echo, we should get the
    # override value we passed to processor_kwargs instead.
    assert num_crops_val == NUM_CROPS_OVERRIDE


def test_processor_with_sad_kwarg_overrides(processor_mock):
    """Ensure that processor kwargs that are unused do not fail."""
    dummy_registry = InputRegistry()
    # Since the processor does not take `does_not_exist` as an arg,
    # it will be filtered, then warn + drop it from the callable
    # to prevent the processor from failing.
    model_config = get_model_config(processor_kwargs={"does_not_exist": 100}, )

    processor = dummy_registry.create_input_processor(model_config)
    num_crops_val = processor(LLMInputs(prompt="foobar"))
    assert num_crops_val == DEFAULT_NUM_CROPS


def test_processor_kwargs_cannot_clobber_reserved_kwargs(processor_mock):
    """Ensure that special kwargs cannot be overridden."""
    dummy_registry = InputRegistry()
    model_config = get_model_config(processor_kwargs={"ctx":
                                                      "something bad"}, )
    processor = dummy_registry.create_input_processor(model_config)
    # It's good enough to make sure this is callable, because if we had
    # an override pushed through, we'd run into issues with multiple
    # values provided for a single argument
    processor(LLMInputs(prompt="foobar"))
