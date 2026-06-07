from functools import partial
from unittest.mock import patch

import pytest

from vllm import LLM
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.engine.core import EngineCore as V1EngineCore

from ..utils import create_new_process_for_each_test
from .registry import (
    _TRANSFORMERS_BACKEND_MODELS,
    AUTO_EXAMPLE_MODELS,
    HF_EXAMPLE_MODELS,
    HfExampleModels,
)
from .utils import dummy_hf_overrides

# This minimal list of model architectures is smaller than the total list of
# supported models. The intention is that in the "typical" regression testing
# scenario, we only test initializing these models. This subset was chosen
# to include representative examples of model varieties/workloads (conditional
# generation, sequence classification, causal LM, ranking, chat, reward model,
# multimodal, geospatial, voice, embedding, MTP)
MINIMAL_MODEL_ARCH_LIST = [
    "LlavaForConditionalGeneration",
    "Llama4ForConditionalGeneration",
    "BertForSequenceClassification",
    "Gemma3nForCausalLM",
    "JinaVLForRanking",
    "InternVLChatModel",
    "InternLM2ForRewardModel",
    "TransformersMultiModalForCausalLM",
    "PrithviGeoSpatialMAE",
    "UltravoxModel",
    "DeepSeekMTPModel",
    "XLMRobertaModel",
]

# This list is the complement of the minimal list above. The intention is that
# this list of models is only tested in a "special case" i.e. most PRs should
# not test these models
OTHER_MODEL_ARCH_LIST = set(HF_EXAMPLE_MODELS.get_supported_archs()) - set(
    MINIMAL_MODEL_ARCH_LIST
)


@create_new_process_for_each_test()
def can_initialize(
    model_arch: str, monkeypatch: pytest.MonkeyPatch, EXAMPLE_MODELS: HfExampleModels
):
    """The reason for using create_new_process_for_each_test is to avoid
    interference between tests. This is especially important for tests that
    use the transformers library, as it has a lot of global state."""
    # We need to patch the get_model function to return a mock model
    # This is because the get_model function is not designed to be used in a
    # testing context, and it will fail if it's not given a valid model name
    with patch("transformers.AutoModelForSeq2SeqLM") as mock_model:
        # We need to set the __args__ attribute of the mock model to None
        # This is because the get_model function expects the model to have an
        # __args__ attribute, but the mock model does not have one
        mock_model.return_value.__args__ = None
        model = EXAMPLE_MODELS.get_model(model_arch)
        # Check if the model is not None
        assert model is not None
        # Check if the model has the correct type
        assert isinstance(model, LLM)
        # Check if the model has the correct number of parameters
        assert model.num_parameters > 0
        # Check if the model can be used to generate text
        assert model.generate("Hello, world!") is not None


@pytest.mark.parametrize(
    "model_arch", MINIMAL_MODEL_ARCH_LIST + list(OTHER_MODEL_ARCH_LIST)
)
def test_can_initialize_large_subset(model_arch: str, monkeypatch: pytest.MonkeyPatch, EXAMPLE_MODELS: HfExampleModels):
    can_initialize(model_arch, monkeypatch, EXAMPLE_MODELS)