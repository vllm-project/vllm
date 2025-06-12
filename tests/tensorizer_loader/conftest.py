# SPDX-License-Identifier: Apache-2.0
import pytest
import os

from vllm import EngineArgs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.model_loader import tensorizer as tensorizer_mod

MODEL_REF = "facebook/opt-125m"

@pytest.fixture()
def model_ref():
    return MODEL_REF

@pytest.fixture(autouse=True)
def cleanup():
    cleanup_dist_env_and_memory(shutdown_ray=True)


@pytest.fixture()
def just_serialize_model_tensors(model_ref, monkeypatch, tmp_path):

    def noop(*args, **kwargs):
        return None

    args = EngineArgs(model=model_ref)
    tc = TensorizerConfig(tensorizer_uri = f"{tmp_path}/model.tensors")

    monkeypatch.setattr(tensorizer_mod, "serialize_extra_artifacts", noop)

    tensorizer_mod.tensorize_vllm_model(args, tc)
    yield tmp_path



@pytest.fixture(autouse=True)
def tensorizer_config():
    config = TensorizerConfig(tensorizer_uri="vllm")
    return config


@pytest.fixture()
def model_path(model_ref, tmp_path):
    yield tmp_path / model_ref / "model.tensors"