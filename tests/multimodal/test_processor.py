from array import array
from typing import Mapping
from unittest.mock import patch

import pytest

from vllm.config import ModelConfig
from vllm.inputs import InputContext, LLMInputs
from vllm.inputs.registry import InputRegistry

from vllm.multimodal import MultiModalRegistry
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData



DUMMY_MODEL_ID = "facebook/opt-125m"
# For processor_kwargs - we test overrides by defining mocks for each place
# it is used, and ensuring that we can pass processor kwargs an override value
# to receive the intended result for things like sequence length etc.
DEFAULT_NUM_CROPS = 4
NUM_CROPS_OVERRIDE = 16

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

# Mocks for all of the places that we use the processor_kwargs
# to override values in different callables
@pytest.fixture
def use_processor_mock():
    """Patches the internal model input processor with an override callable."""
    def custom_processor(ctx: InputContext,
                         llm_inputs: LLMInputs,
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
        seq_data = SequenceData(array(VLLM_TOKEN_ID_ARRAY_TYPE, [0] * num_crops))
        return seq_data, None

    with patch("vllm.inputs.registry.InputRegistry._default_dummy_data_factory",
               custom_dummy_data_factory):
        yield


### Test for default processor logic & processor_kwargs wrapping
def test_default_processor_is_a_noop():
    """Ensure that by default, there is no processor override."""
    dummy_registry = InputRegistry()
    model_config = get_model_config()
    processor = dummy_registry.create_input_processor(model_config)
    proc_inputs = LLMInputs(prompt_token_ids=[], prompt="")
    proc_outputs = processor(inputs=proc_inputs)
    assert proc_inputs is proc_outputs

@pytest.mark.parametrize("num_crops", [None, NUM_CROPS_OVERRIDE])
def test_processor_default_kwargs(use_processor_mock, num_crops):
    """Ensure that we can override processor kwargs."""
    dummy_registry = InputRegistry()
    # If we have a value for num_crops, pass the override value and make
    # sure we get that value as a return-value from out mock processor,
    # otherwise fall back to the default value
    processor_kwargs = None if num_crops is None else {"num_crops": num_crops}
    expected_num_crops = DEFAULT_NUM_CROPS if num_crops is None else num_crops
    model_config = get_model_config(processor_kwargs=processor_kwargs)
    processor = dummy_registry.create_input_processor(model_config)

    num_crops_val = processor(LLMInputs(prompt_token_ids=[], prompt=""))
    assert num_crops_val == expected_num_crops


@pytest.mark.parametrize("processor_kwargs", 
    [
        {"does_not_exist": 100}, # Not part of the signature
        {"ctx": "something bad"} # Part of the signature, not keyword only
    ]
)
def test_processor_with_sad_kwarg_overrides(use_processor_mock,
                                            processor_kwargs):
    """Ensure invalid processor_kwargs can't be used in the input processor."""
    dummy_registry = InputRegistry()

    model_config = get_model_config(processor_kwargs=processor_kwargs)

    processor = dummy_registry.create_input_processor(model_config)
    num_crops_val = processor(LLMInputs(prompt_token_ids=[], prompt=""))
    assert num_crops_val == DEFAULT_NUM_CROPS


### Test overrides for the dummy data
@pytest.mark.parametrize("num_crops", [None, NUM_CROPS_OVERRIDE])
def test_dummy_data_kwarg_overrides(use_dummy_data_mock, num_crops):
    processor_kwargs = None if num_crops is None else {"num_crops": num_crops}
    expected_seq_count = DEFAULT_NUM_CROPS if num_crops is None else num_crops
    dummy_registry = InputRegistry()
    model_config = get_model_config(
        processor_kwargs=processor_kwargs,
    )
    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(model_config)

    # NOTE: seq_len is thrown away here since this will leverage the
    # default dummy data factory that we have patched in, whose seq
    # len is solely dependent on the value of the processor_kwargs.
    seq_data, _ = dummy_registry.dummy_data_for_profiling(
        model_config,
        seq_len=-1,
        mm_registry=mm_registry
    )
    assert len(seq_data.prompt_token_ids) == expected_seq_count


@pytest.mark.parametrize("processor_kwargs", 
    [
        {"does_not_exist": 100}, # Not part of the signature
        {"ctx": "something bad"} # Part of the signature, not keyword only
    ]
)
def test_dummy_data_with_sad_kwarg_overrides(use_dummy_data_mock, processor_kwargs):
    """Ensure that dummy_data kwargs that are unused do not fail."""
    dummy_registry = InputRegistry()
    model_config = get_model_config(
        processor_kwargs=processor_kwargs,
    )
    mm_registry = MultiModalRegistry()
    mm_registry.init_mm_limits_per_prompt(model_config)

    # NOTE: seq_len is thrown away here since this will leverage the
    # default dummy data factory that we have patched in, whose seq
    # len is solely dependent on the value of the processor_kwargs.
    seq_data, _ = dummy_registry.dummy_data_for_profiling(
        model_config,
        seq_len=-1,
        mm_registry=mm_registry
    )
    assert len(seq_data.prompt_token_ids) == DEFAULT_NUM_CROPS
