# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AsyncTP numerical coverage kept outside the shared distributed suite."""

import pytest
import torch

from tests.compile.passes.distributed.test_async_tp import (
    TestAGMMModel,
    TestMMRSModel,
    async_tp_pass_on_test_model,
)
from tests.utils import multi_gpu_test
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_file_store_init_method


class TestRocmMMRSModel(TestMMRSModel):
    def forward(self, hidden_states):
        view = hidden_states.reshape(-1, self.hidden_size)
        mm = torch.ops.vllm.rocm_unquantized_gemm(view, self.gate_proj, None)
        return tensor_model_parallel_reduce_scatter(mm, dim=0)

    def ops_in_model_before(self):
        return super().ops_in_model_before() + [
            torch.ops.vllm.rocm_unquantized_gemm.default
        ]


class TestRocmAGMMModel(TestAGMMModel):
    def __init__(self, hidden_size=16, dtype=torch.bfloat16):
        super().__init__(hidden_size, dtype)
        self.weight = torch.nn.Parameter(
            torch.empty((hidden_size * 2, hidden_size)), requires_grad=False
        )
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        view = hidden_states.reshape(-1, self.hidden_size)
        gathered = tensor_model_parallel_all_gather(view, dim=0)
        return torch.ops.vllm.rocm_unquantized_gemm(gathered, self.weight, None)

    def ops_in_model_before(self):
        return super().ops_in_model_before() + [
            torch.ops.vllm.rocm_unquantized_gemm.default
        ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend contract")
@pytest.mark.parametrize(
    "test_model", [TestMMRSModel, TestAGMMModel, TestRocmMMRSModel, TestRocmAGMMModel]
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_rocm_async_tp_bf16_rewrite_and_correctness(test_model, dynamic):
    """Real symmetric-memory AG/GEMM and GEMM/RS must replace the collectives."""
    torch.multiprocessing.spawn(
        async_tp_pass_on_test_model,
        args=(
            2,
            test_model,
            8,
            16,
            128,
            torch.bfloat16,
            dynamic,
            get_file_store_init_method(),
            True,
        ),
        nprocs=2,
    )
