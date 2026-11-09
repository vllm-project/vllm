# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


@pytest.fixture
def fp32_cublas_reduction():
    """Make cuBLAS reduce BF16 split-K partial sums in FP32 during the test.

    Set and restored by hand: cuBLASModule keeps the flag outside its
    `__dict__`, so monkeypatch (pytest 9.2 on) records it as absent and
    fails deleting it on undo."""
    matmul = torch.backends.cuda.matmul
    reduced = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    yield
    matmul.allow_bf16_reduced_precision_reduction = reduced
