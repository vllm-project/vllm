# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import pickle
from typing import Any

import pytest
import requests
import torch
from torch.multiprocessing.reductions import reduce_tensor

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

def _encode(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def _decode(obj: str) -> Any:
    return pickle.loads(base64.b64decode(obj.encode('utf-8')))


class TestWorkerExtension:
    def get_model_name(self) -> str:
        """Test non-pydantic return type."""
        return MODEL_NAME

    def echo_args_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        """Echo back both args and kwargs."""
        return dict(
            args=list(args),
            kwargs=kwargs,
            total_items=len(args) + len(kwargs),
        )
    
    def get_tensor_meta(self, marshal_obj: str) -> str:
        ipc_handle = _decode(marshal_obj)
        func, args = ipc_handle
        args = list(args)
        args[6] = 0 # set device to 0
        tensor: torch.Tensor = func(*args)
        return _encode((tensor.shape, tensor.dtype))

    def return_none(self, *args, **kwargs) -> None:
        """Test method that does not return anything"""
        return


@pytest.fixture(scope="module")
def server():
    args = [
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--worker-extension-cls",
        "tests.entrypoints.openai.test_collective_rpc.TestWorkerExtension",
    ]
    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={"VLLM_SERVER_DEV_MODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
    ) as remote_server:
        yield remote_server


def test_get_model_name(server):
    """Test basic response"""
    response = requests.post(
        server.url_for("collective_rpc"), json={"method": "get_model_name"}
    )
    assert response.status_code == 200
    results = response.json()
    assert "results" in results
    assert results["results"] == [MODEL_NAME]


def test_return_none(server):
    """Test return none"""
    response = requests.post(
        server.url_for("collective_rpc"), json={"method": "return_none"}
    )
    assert response.status_code == 200
    results = response.json()
    assert results["results"] == [None]


def test_echo_args_kwargs(server):
    """Test args, kwargs, and dict response"""
    args = ["arg1", "arg2"]
    kwargs = {"key1": "value1", "key2": "value2"}
    response = requests.post(
        server.url_for("collective_rpc"),
        json={"method": "echo_args_kwargs", "args": args, "kwargs": kwargs},
    )
    assert response.status_code == 200
    results = response.json()
    result = results["results"][0]
    assert result["args"] == args
    assert result["kwargs"] == kwargs
    assert result["total_items"] == len(args) + len(kwargs)


def test_get_tensor_meta(server):
    """Test args, kwargs, and dict response"""
    tensor = torch.randn(10, 10, device='cuda')
    handle = reduce_tensor(tensor)
    base64_ipc_handle = _encode(handle)
    args = [base64_ipc_handle]
    response = requests.post(server.url_for("collective_rpc"),
                             json={
                                 "method": "get_tensor_meta",
                                 "args": args,
                             })
    assert response.status_code == 200
    results = response.json()
    result = results["results"][0]
    assert _decode(result) == (tensor.shape, tensor.dtype)
