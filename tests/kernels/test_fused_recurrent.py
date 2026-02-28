# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_recurrent requires CUDA or ROCm",
)

PAD_SLOT_ID = -1

BATCH_SIZES = [1, 2, 4]
SEQ_LENS = [1, 4, 16]
NUM_PADDINGS = [0, 1, 2, 4]
DTYPES = [torch.float32, torch.bfloat16]
MAX_BATCH_SIZES = [4, 8, 16]
ACTUAL_BATCHES = [1, 2, 4]
TP_SIZES = [1, 2, 4, 8]

# Qwen3-Next-80B shapes: (H, HV, K, V) for different TP sizes
QWEN3_NEXT_SHAPES = {
    1: (16, 32, 128, 128),
    2: (8, 16, 128, 128),
    4: (4, 8, 128, 128),
    8: (2, 4, 128, 128),
}


def get_fused_recurrent_fn():
    """Import the fused_recurrent function, skip if not available."""
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )

        return fused_recurrent_gated_delta_rule
    except ImportError:
        pytest.skip("fused_recurrent_gated_delta_rule not available")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.parametrize("num_padding", NUM_PADDINGS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@torch.inference_mode()
def test_fused_recurrent_pad_slot_id(
    batch_size: int,
    num_padding: int,
    seq_len: int,
    dtype: torch.dtype,
):
    """Test kernel handles PAD_SLOT_ID (-1) without crashing (Issue #31186)."""
    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")

    H, HV, K, V = 4, 4, 64, 64
    total_seqs = batch_size + num_padding

    valid_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    if num_padding > 0:
        padding_indices = torch.full(
            (num_padding,), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        ssm_state_indices = torch.cat([valid_indices, padding_indices])
    else:
        ssm_state_indices = valid_indices

    q = torch.randn(total_seqs, seq_len, H, K, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(total_seqs, seq_len, H, K, dtype=dtype, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(total_seqs, seq_len, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(total_seqs, seq_len, HV, dtype=dtype, device=device))
    beta = torch.rand(total_seqs, seq_len, HV, dtype=dtype, device=device).sigmoid()
    initial_state = torch.zeros(batch_size, HV, K, V, dtype=dtype, device=device)

    try:
        output, final_state = fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial_state,
            inplace_final_state=True,
            ssm_state_indices=ssm_state_indices,
        )
        assert output.shape == (total_seqs, seq_len, HV, V)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    except RuntimeError as e:
        if "illegal memory access" in str(e).lower():
            pytest.fail(f"Kernel crashed with PAD_SLOT_ID - regression #31186: {e}")
        raise


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@torch.inference_mode()
def test_fused_recurrent_all_padding(batch_size: int):
    """Test edge case where all indices are PAD_SLOT_ID."""
    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.float32

    H, HV, K, V = 4, 4, 64, 64
    seq_len = 1

    ssm_state_indices = torch.full(
        (batch_size,), PAD_SLOT_ID, dtype=torch.int64, device=device
    )

    q = torch.randn(batch_size, seq_len, H, K, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(batch_size, seq_len, H, K, dtype=dtype, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(batch_size, seq_len, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(batch_size, seq_len, HV, dtype=dtype, device=device))
    beta = torch.rand(batch_size, seq_len, HV, dtype=dtype, device=device).sigmoid()
    initial_state = torch.zeros(1, HV, K, V, dtype=dtype, device=device)

    try:
        output, _ = fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial_state,
            inplace_final_state=True,
            ssm_state_indices=ssm_state_indices,
        )
        assert output.shape == (batch_size, seq_len, HV, V)
    except RuntimeError as e:
        if "illegal memory access" in str(e).lower():
            pytest.fail(f"Kernel crashed with all PAD_SLOT_ID: {e}")
        raise


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@torch.inference_mode()
def test_fused_recurrent_basic_forward(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
):
    """Test basic forward pass."""
    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")

    H, HV, K, V = 4, 4, 64, 64

    q = torch.randn(batch_size, seq_len, H, K, dtype=dtype, device=device) * 0.1
    k = F.normalize(
        torch.randn(batch_size, seq_len, H, K, dtype=dtype, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(batch_size, seq_len, HV, V, dtype=dtype, device=device) * 0.1
    g = F.logsigmoid(torch.randn(batch_size, seq_len, HV, dtype=dtype, device=device))
    beta = torch.rand(batch_size, seq_len, HV, dtype=dtype, device=device).sigmoid()
    initial_state = torch.zeros(batch_size, HV, K, V, dtype=dtype, device=device)

    ssm_state_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    ssm_state_indices = ssm_state_indices.unsqueeze(1).expand(batch_size, seq_len)
    ssm_state_indices = ssm_state_indices.contiguous()

    output, final_state = fused_recurrent(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state,
        inplace_final_state=True,
        ssm_state_indices=ssm_state_indices,
    )

    assert output.shape == (batch_size, seq_len, HV, V)
    assert output.dtype == dtype
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert final_state.shape == (batch_size, HV, K, V)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_recurrent_dtype_support(dtype: torch.dtype):
    """Test kernel supports various dtypes."""
    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")

    B, T, H, HV, K, V = 2, 4, 4, 4, 64, 64

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, HV, dtype=dtype, device=device))
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    initial_state = torch.zeros(B, HV, K, V, dtype=dtype, device=device)

    output, _ = fused_recurrent(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=initial_state,
        ssm_state_indices=None,
    )

    assert output.dtype == dtype


@pytest.mark.parametrize("actual_batch", ACTUAL_BATCHES)
@pytest.mark.parametrize("max_batch_size", MAX_BATCH_SIZES)
@torch.inference_mode()
def test_fused_recurrent_cuda_graph_scenario(
    max_batch_size: int,
    actual_batch: int,
):
    """Test CUDA Graph scenario that caused Issue #31186."""
    if actual_batch > max_batch_size:
        pytest.skip("actual_batch must be <= max_batch_size")

    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    H, HV, K, V = 8, 8, 128, 128
    seq_len = 1

    initial_state = torch.zeros(max_batch_size, HV, K, V, dtype=dtype, device=device)

    valid_indices = torch.arange(actual_batch, dtype=torch.int64, device=device)
    padding_count = max_batch_size - actual_batch
    if padding_count > 0:
        padding_indices = torch.full(
            (padding_count,), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        ssm_state_indices = torch.cat([valid_indices, padding_indices])
    else:
        ssm_state_indices = valid_indices

    q = torch.randn(max_batch_size, seq_len, H, K, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(max_batch_size, seq_len, H, K, dtype=dtype, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(max_batch_size, seq_len, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(
        torch.randn(max_batch_size, seq_len, HV, dtype=dtype, device=device)
    )
    beta = torch.rand(max_batch_size, seq_len, HV, dtype=dtype, device=device).sigmoid()

    try:
        output, final_state = fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial_state,
            inplace_final_state=True,
            ssm_state_indices=ssm_state_indices,
        )

        assert output.shape == (max_batch_size, seq_len, HV, V)
        valid_output = output[:actual_batch]
        assert not torch.isnan(valid_output).any()
        assert not torch.isinf(valid_output).any()

    except RuntimeError as e:
        if "illegal memory access" in str(e).lower():
            pytest.fail(f"CUDA Graph simulation crashed - regression #31186: {e}")
        raise


@pytest.mark.parametrize("actual_batch", [1, 4])
@pytest.mark.parametrize("max_batch_size", [8, 16])
@pytest.mark.parametrize("tp_size", TP_SIZES)
@torch.inference_mode()
def test_fused_recurrent_qwen3_next_shapes(
    tp_size: int,
    max_batch_size: int,
    actual_batch: int,
):
    """Test with Qwen3-Next-80B shapes and PAD_SLOT_ID padding."""
    if actual_batch > max_batch_size:
        pytest.skip("actual_batch must be <= max_batch_size")

    fused_recurrent = get_fused_recurrent_fn()
    set_random_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    H, HV, K, V = QWEN3_NEXT_SHAPES[tp_size]
    seq_len = 1

    valid_indices = torch.arange(actual_batch, dtype=torch.int64, device=device)
    padding_count = max_batch_size - actual_batch
    if padding_count > 0:
        padding_indices = torch.full(
            (padding_count,), PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        ssm_state_indices = torch.cat([valid_indices, padding_indices])
    else:
        ssm_state_indices = valid_indices

    q = torch.randn(max_batch_size, seq_len, H, K, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(max_batch_size, seq_len, H, K, dtype=dtype, device=device),
        p=2,
        dim=-1,
    )
    v = torch.randn(max_batch_size, seq_len, HV, V, dtype=dtype, device=device)
    g = F.logsigmoid(
        torch.randn(max_batch_size, seq_len, HV, dtype=dtype, device=device)
    )
    beta = torch.rand(max_batch_size, seq_len, HV, dtype=dtype, device=device).sigmoid()
    initial_state = torch.zeros(max_batch_size, HV, K, V, dtype=dtype, device=device)

    try:
        output, final_state = fused_recurrent(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=K**-0.5,
            initial_state=initial_state,
            inplace_final_state=True,
            ssm_state_indices=ssm_state_indices,
        )

        assert output.shape == (max_batch_size, seq_len, HV, V)
        valid_output = output[:actual_batch]
        assert not torch.isnan(valid_output).any()
        assert not torch.isinf(valid_output).any()

    except RuntimeError as e:
        if "illegal memory access" in str(e).lower():
            pytest.fail(f"Kernel crashed with Qwen3-Next shapes (TP={tp_size}): {e}")
        raise
