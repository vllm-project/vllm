# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Value-path range handling in the TurboQuant store kernel.

Regression coverage for gh-53334 observation 2: the per-vector value scale and
zero point are stored as fp16. A bf16 value outlier whose magnitude exceeds the
fp16 finite range (65504) used to cast to -inf at store time, so every element
of that vector reconstructed as inf.
"""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

DEVICE_TYPE = current_platform.device_type

D = 128
H = 1
N = 1
KEY_PACKED_SIZE = D
BLOCK_SIZE = 16
NUM_BLOCKS = 1


def _store_and_reconstruct(outlier: float, value_quant_bits: int):
    """Store one value vector containing `outlier`; reconstruct it from cache."""
    val_data_bytes = D * value_quant_bits // 8
    slot_bytes = KEY_PACKED_SIZE + val_data_bytes + 4

    value = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)
    value[0, 0, 1:] = torch.linspace(-1.0, 4.0, D - 1, device=DEVICE_TYPE).to(
        torch.bfloat16
    )
    value[0, 0, 0] = outlier
    key = torch.zeros(N, H, D, dtype=torch.bfloat16, device=DEVICE_TYPE)

    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, H, slot_bytes, dtype=torch.uint8, device=DEVICE_TYPE
    )
    triton_turboquant_store(
        key,
        value,
        kv_cache,
        torch.zeros(N, dtype=torch.int32, device=DEVICE_TYPE),
        torch.eye(D, dtype=torch.float32, device=DEVICE_TYPE),
        torch.zeros(1, dtype=torch.float32, device=DEVICE_TYPE),
        mse_bits=1,
        key_packed_size=KEY_PACKED_SIZE,
        value_quant_bits=value_quant_bits,
        key_fp8=True,
    )
    torch.cuda.synchronize()

    slot = kv_cache[0, 0, 0].cpu()
    meta = slot[KEY_PACKED_SIZE + val_data_bytes :][:4].numpy().tobytes()
    scale = np.frombuffer(meta[0:2], dtype=np.float16)[0].astype(np.float32)
    zero = np.frombuffer(meta[2:4], dtype=np.float16)[0].astype(np.float32)

    packed = slot[KEY_PACKED_SIZE : KEY_PACKED_SIZE + val_data_bytes].numpy()
    q = np.zeros(D, dtype=np.float32)
    if value_quant_bits == 4:
        q[0::2] = packed & 0x0F
        q[1::2] = (packed >> 4) & 0x0F
    else:  # 3-bit, 8 values packed into 3 bytes
        bits = np.unpackbits(packed, bitorder="little").reshape(-1, 24)
        for i in range(8):
            grp = bits[:, i * 3 : (i + 1) * 3]
            q[i::8] = (grp * (1 << np.arange(3))).sum(axis=1)[: D // 8]

    return q * scale + zero, value[0, 0].float().cpu().numpy(), scale, zero


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("value_quant_bits", [4])
@pytest.mark.parametrize(
    "outlier",
    [
        -125000.0,  # beyond fp16 range; measured on a real 27B v-cache sink
        -70000.0,  # just beyond fp16 range
        -42000.0,  # within fp16 range (already worked before the fix)
        -1.0,  # ordinary vector
    ],
)
def test_value_outlier_reconstructs_finite(outlier, value_quant_bits):
    """No value magnitude may reconstruct as inf.

    Before the fix, |outlier| > 65504 stored a -inf zero point and every
    element of the vector came back as inf.
    """
    recon, ref, scale, zero = _store_and_reconstruct(outlier, value_quant_bits)

    assert np.isfinite(scale), f"stored scale is not finite: {scale}"
    assert np.isfinite(zero), f"stored zero point is not finite: {zero}"
    assert np.isfinite(recon).all(), (
        f"outlier {outlier} poisoned the vector: "
        f"{np.count_nonzero(~np.isfinite(recon))}/{D} elements non-finite"
    )

    # The elements that are not the outlier must stay usable. The bound is the
    # inherent 4-bit per-vector quantization step for a vector with this range,
    # which is what an in-range outlier of the same magnitude already produces.
    step = (ref.clip(-65504.0).max() - max(ref.min(), -65504.0)) / 15.0
    assert np.abs(recon[1:] - ref[1:]).max() <= step + 1e-3
