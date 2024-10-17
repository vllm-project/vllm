"""Tests for HF_HUB_OFFLINE mode"""
import importlib
import sys
import weakref

import pytest

from vllm import LLM

from ...conftest import cleanup

MODEL_CONFIGS = [
    {
        "model": "facebook/opt-125m",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.10,
        "max_num_batched_tokens": 4096,
        "tensor_parallel_size": 1,
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.50,
        "max_model_len": 2048,
        "max_num_batched_tokens": 4096,
        "tensor_parallel_size": 1,
        "tokenizer_mode": "mistral",
    },
]


@pytest.fixture(scope="function")
def llm(model_config: dict):
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(**model_config)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup()


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("model_config", MODEL_CONFIGS)
def test_offline_mode(model_config: dict, llm: LLM, monkeypatch):
    # we use the llm fixture to ensure the model files are in-cache
    del llm

    # Set HF to offline mode and ensure we can still construct an LLM
    try:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        # Need to re-import huggingface_hub and friends to setup offline mode
        _re_import_modules()
        # Cached model files should be used in offline mode
        LLM(**model_config)
    finally:
        # Reset the environment after the test
        # NB: Assuming tests are run in online mode
        monkeypatch.delenv("HF_HUB_OFFLINE")
        _re_import_modules()
        pass


def _re_import_modules():
    hf_hub_module_names = [
        k for k in sys.modules if k.startswith("huggingface_hub")
    ]
    transformers_module_names = [
        k for k in sys.modules if k.startswith("transformers")
        and not k.startswith("transformers_modules")
    ]

    reload_exception = None
    for module_name in hf_hub_module_names + transformers_module_names:
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            reload_exception = e
            # Try to continue clean up so that other tests are less likely to
            # be affected

    # Error this test if reloading a module failed
    if reload_exception is not None:
        raise reload_exception
