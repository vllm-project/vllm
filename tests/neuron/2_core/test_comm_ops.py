# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Callable
from unittest.mock import patch

import pytest
import torch
import torch_xla.distributed.xla_multiprocessing as xmp
from typing_extensions import ParamSpec

from vllm.distributed.communication_op import (
    tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.utils import get_distributed_init_method, get_open_port

_P = ParamSpec("_P")


def reinitialize_neuron_runtime(f: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to reinitialize the Neuron Runtime before executing a test.
    This is necessary for distributed tests which need to reallocate Neuron
    Cores to separate subprocesses.
    """

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        runtime = torch.classes.neuron.Runtime()
        runtime.initialize()
        runtime.unsafe_close()

        f(*args, **kwargs)
        runtime.initialize()

    return wrapper


def all_gather_test_worker(index, tp_degree, distributed_init_method):
    init_distributed_environment(tp_degree,
                                 index,
                                 distributed_init_method,
                                 index,
                                 backend="xla")
    ensure_model_parallel_initialized(tp_degree, 1)

    num_dimensions = 3
    tensor_size = list(range(2, num_dimensions + 2))
    total_size = 1
    for s in tensor_size:
        total_size *= s

    all_gather_dimension = -1
    all_tensors = [
        torch.arange(total_size, dtype=torch.float32,
                     device="xla").reshape(tensor_size) * (r + 1)
        for r in range(tp_degree)
    ]
    expected = torch.cat(all_tensors, dim=all_gather_dimension)
    t = all_tensors[index % tp_degree]
    t = tensor_model_parallel_all_gather(t, all_gather_dimension)
    torch.testing.assert_close(t, expected)


def all_reduce_test_worker(index, tp_degree, distributed_init_method):
    init_distributed_environment(tp_degree,
                                 index,
                                 distributed_init_method,
                                 index,
                                 backend="xla")
    ensure_model_parallel_initialized(tp_degree, 1)

    num_elements = 8
    all_tensors = [
        torch.arange(num_elements, dtype=torch.float32, device="xla") * (r + 1)
        for r in range(tp_degree)
    ]
    expected = torch.sum(torch.stack(all_tensors, dim=0), dim=0)
    t = all_tensors[index % tp_degree]
    t = tensor_model_parallel_all_reduce(t)
    torch.testing.assert_close(t, expected)


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("test_target",
                         [all_reduce_test_worker, all_gather_test_worker])
@reinitialize_neuron_runtime
def test_neuron_multi_process_tensor_parallel(monkeypatch, tp_size,
                                              test_target):

    with patch('torch_xla._XLAC._xla_runtime_is_initialized',
               return_value=False):
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())

        monkeypatch.setenv("VLLM_USE_V1", "1")
        monkeypatch.setenv("NEURONCORE_NUM_DEVICES", str(tp_size))
        monkeypatch.setenv("NEURON_PJRT_PROCESSES_NUM_DEVICES",
                           ','.join(['1' for _ in range(tp_size)]))

        xmp.spawn(test_target, args=(tp_size, distributed_init_method))
