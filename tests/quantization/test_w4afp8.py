# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the ``w4afp8`` layout translation.

Run `pytest tests/quantization/test_w4afp8.py`.

These tests cover the on-disk repack (signed INT4 -> ``uint4b8``) and the
configuration wiring. Whether the W4A8 kernel consumes the resulting tensor
correctly needs a GPU end-to-end run and is out of scope here.
"""

import sys

import pytest
import torch

from vllm.model_executor.layers.quantization.w4afp8 import (
    W4AFP8Config,
    _repack_signed_int4_to_uint4b8,
)


def _pack_nibbles(nibbles: list[int]) -> torch.Tensor:
    """Pack 4-bit patterns two per byte, low nibble first."""
    assert len(nibbles) % 2 == 0
    payload = [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, len(nibbles), 2)]
    return torch.tensor([payload], dtype=torch.uint8).view(torch.int8)


def _unpack_nibbles(bytes_: torch.Tensor) -> list[int]:
    """Inverse of `_pack_nibbles` for a byte view of the repacked tensor."""
    return [
        nibble
        for byte in bytes_.flatten().tolist()
        for nibble in (byte & 0xF, (byte >> 4) & 0xF)
    ]


def test_repack_known_bytes():
    # int4 payloads -> packed bytes 0x88, 0xF1, 0x7F, 0x00 (one int32 word).
    packed = _pack_nibbles([0x8, 0x8, 0x1, 0xF, 0xF, 0x7, 0x0, 0x0])

    out = _repack_signed_int4_to_uint4b8(packed)

    assert out.dtype is torch.int32
    assert out.shape == (1, packed.shape[-1] // 4)
    words = out.view(torch.uint8).reshape(-1, 4)
    if sys.byteorder == "big":
        words = words.flip(-1)
    assert words.flatten().tolist() == [0x00, 0x79, 0xF7, 0x88]


def test_repack_shifts_every_nibble_by_eight():
    nibbles = list(range(16)) * 2  # all 16 int4 payloads, 32 nibbles = 4 words

    out = _repack_signed_int4_to_uint4b8(_pack_nibbles(nibbles))

    # A word holds 4 consecutive bytes; undo the per-word byte order so the
    # assertion does not depend on the host's endianness.
    words = out.view(torch.uint8).reshape(-1, 4)
    if sys.byteorder == "big":
        words = words.flip(-1)
    assert _unpack_nibbles(words) == [(value + 8) % 16 for value in nibbles]


def test_repack_leaves_input_untouched():
    packed = _pack_nibbles([0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8])
    before = packed.clone()

    _repack_signed_int4_to_uint4b8(packed)

    assert torch.equal(packed, before)


def test_repack_rejects_non_int8():
    with pytest.raises(AssertionError):
        _repack_signed_int4_to_uint4b8(torch.zeros(1, 4, dtype=torch.int32))


def test_repack_rejects_partial_word():
    with pytest.raises(AssertionError):
        _repack_signed_int4_to_uint4b8(torch.zeros(1, 6, dtype=torch.int8))


def test_config_name_and_dense_dispatch():
    assert W4AFP8Config.get_name() == "w4afp8"

    config = W4AFP8Config.from_config(
        {
            "quant_method": "w4afp8",
            "group_size": 128,
            "modules_to_not_convert": ["lm_head"],
        }
    )

    # Dense layers must go through the native FP8 block-128 path, because these
    # checkpoints name the block scale `weight_scale_inv`.
    dense = config._dense_fp8_config
    assert dense.is_checkpoint_fp8_serialized
    assert dense.activation_scheme == "dynamic"
    assert dense.weight_block_size == [128, 128]
    assert dense.ignored_layers == ["lm_head"]


def test_config_rejects_other_quant_method():
    with pytest.raises(ValueError, match="w4afp8"):
        W4AFP8Config.from_config({"quant_method": "compressed-tensors"})
