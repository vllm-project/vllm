# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HF_HUB_OFFLINE mode."""

import importlib
import json
import os
import subprocess
import sys
import textwrap

import pytest
import regex as re
import urllib3

from vllm.transformers_utils.repo_utils import hf_api

MODEL_CONFIGS = [
    {
        "model": "facebook/opt-125m",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.20,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
    },
    {
        "model": "Qwen/Qwen3-0.6B",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.50,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
        "tokenizer": "Qwen/Qwen3-4B",
    },
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "enforce_eager": True,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
        "tokenizer_mode": "mistral",
    },
    # TODO: re-enable once these tests are run with V1
    # {
    #     "model": "sentence-transformers/all-MiniLM-L12-v2",
    #     "enforce_eager": True,
    #     "gpu_memory_utilization": 0.20,
    #     "max_model_len": 64,
    #     "max_num_batched_tokens": 64,
    #     "max_num_seqs": 64,
    #     "tensor_parallel_size": 1,
    # },
]


def _create_runner(vllm_runner, model_config):
    runner_config = model_config.copy()
    model = runner_config.pop("model")
    tokenizer_name = runner_config.pop("tokenizer", None)
    return vllm_runner(model, tokenizer_name=tokenizer_name, **runner_config)


def test_local_model_with_modelscope_preference():
    """A local checkpoint generates identical tokens without optional Hub packages."""
    model = hf_api().snapshot_download(repo_id="facebook/opt-125m")
    token_ids = []
    for use_modelscope in ("false", "true"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import json
                    import sys
                    sys.modules["modelscope"] = None
                    from vllm import LLM, SamplingParams

                    llm = LLM(
                        model=sys.argv[1], enforce_eager=True, seed=0,
                        max_model_len=64, max_num_batched_tokens=64,
                        max_num_seqs=1, gpu_memory_utilization=0.2,
                    )
                    output = llm.generate(
                        ["The capital of France is"],
                        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                    )
                    ids = output[0].outputs[0].token_ids
                    assert len(ids) == 8
                    print("TOKEN_IDS=" + json.dumps(ids))
                    """),
                model,
            ],
            env={
                **os.environ,
                "VLLM_USE_MODELSCOPE": use_modelscope,
                "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_OFFLINE": "1",
            },
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        token_ids.append(
            json.loads(
                next(
                    line.removeprefix("TOKEN_IDS=")
                    for line in result.stdout.splitlines()
                    if line.startswith("TOKEN_IDS=")
                )
            )
        )
    assert token_ids[0] == token_ids[1]


@pytest.fixture(scope="module")
def cache_models(vllm_runner):
    # Cache model files first
    for model_config in MODEL_CONFIGS:
        with _create_runner(vllm_runner, model_config):
            pass

    yield


@pytest.mark.skip_global_cleanup
@pytest.mark.usefixtures("cache_models")
def test_offline_mode(monkeypatch: pytest.MonkeyPatch, vllm_runner):
    # Set HF to offline mode and ensure we can still construct an LLM
    with monkeypatch.context() as m:
        try:
            m.setenv("HF_HUB_OFFLINE", "1")
            m.setenv("VLLM_NO_USAGE_STATS", "1")

            def disable_connect(*args, **kwargs):
                raise RuntimeError("No http calls allowed")

            m.setattr(
                urllib3.connection.HTTPConnection,
                "connect",
                disable_connect,
            )
            m.setattr(
                urllib3.connection.HTTPSConnection,
                "connect",
                disable_connect,
            )

            # Need to re-import huggingface_hub
            # and friends to set up offline mode
            _re_import_modules()
            # Cached model files should be used in offline mode
            for model_config in MODEL_CONFIGS:
                with _create_runner(vllm_runner, model_config):
                    pass
        finally:
            # Reset the environment after the test
            # NB: Assuming tests are run in online mode
            _re_import_modules()


def _re_import_modules():
    hf_hub_module_names = [k for k in sys.modules if k.startswith("huggingface_hub")]
    transformers_module_names = [
        k
        for k in sys.modules
        if k.startswith("transformers") and not k.startswith("transformers_modules")
    ]

    # These modules are aliased in Transformers v5 and so cannot be reloaded directly
    aliased_module_patterns = [
        r".+\.tokenization_utils$",
        r".+\.tokenization_utils_fast$",
        r".+\.image_processing_utils_fast$",
        r".+\.models\..+\.image_processing_.+_fast$",
    ]

    reload_exception = None
    for module_name in hf_hub_module_names + transformers_module_names:
        if any(re.match(pattern, module_name) for pattern in aliased_module_patterns):
            # Remove from sys.modules so they are re-aliased on next import
            del sys.modules[module_name]
            continue
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            reload_exception = e
            # Try to continue clean up so that other tests are less likely to
            # be affected

    # Error this test if reloading a module failed
    if reload_exception is not None:
        raise reload_exception


@pytest.mark.skip_global_cleanup
@pytest.mark.usefixtures("cache_models")
def test_model_from_huggingface_offline(monkeypatch: pytest.MonkeyPatch, vllm_runner):
    # Set HF to offline mode and ensure we can still construct an LLM
    with monkeypatch.context() as m:
        try:
            m.setenv("HF_HUB_OFFLINE", "1")
            m.setenv("VLLM_NO_USAGE_STATS", "1")

            def disable_connect(*args, **kwargs):
                raise RuntimeError("No http calls allowed")

            m.setattr(
                urllib3.connection.HTTPConnection,
                "connect",
                disable_connect,
            )
            m.setattr(
                urllib3.connection.HTTPSConnection,
                "connect",
                disable_connect,
            )
            # Need to re-import huggingface_hub
            # and friends to set up offline mode
            _re_import_modules()
            with vllm_runner("facebook/opt-125m"):
                pass
        finally:
            # Reset the environment after the test
            # NB: Assuming tests are run in online mode
            _re_import_modules()
