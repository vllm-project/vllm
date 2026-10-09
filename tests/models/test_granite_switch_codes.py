# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden-vector tests for the Granite Switch Kerdock/DG codebook.

Granite Switch checkpoints are produced by an out-of-tree composer that embeds
adapter control tokens whose addresses are decoded here. If this construction
ever drifts from the composer's, routing silently selects the wrong adapter --
no exception, no stack trace, just wrong output. The digests below therefore
pin the exact bytes of every codebook the model can build, and the same
digests are asserted on the composer side.

Regenerating these constants is only correct when the construction is
*intentionally* changed, and requires the composer to be updated in lockstep.

CPU only; no GPU or network access required.
"""

import hashlib
from dataclasses import dataclass

import pytest
import torch

from vllm.model_executor.models.granite_switch_utils import (
    KerdockDGCodeGenerator,
    recover_count_from_signal,
)


@dataclass(frozen=True)
class GoldenCodebook:
    capacity: int
    dim: int
    sha256: str
    # Sign pattern of a few spread-out addresses, as one big-endian integer per
    # address: bit i (counting from the most significant) is set when entry i of
    # the code vector is negative. Present so that a digest mismatch can be
    # localised to a specific address instead of just failing.
    addresses: tuple[int, ...]
    sign_bits: tuple[int, ...]


GOLDEN: dict[tuple[int, str], GoldenCodebook] = {
    (6, "kerdock"): GoldenCodebook(
        capacity=2048,
        dim=64,
        sha256="e0f70efdf1e8bf7da601b07d0affe664c4f66fc7e750914ef145e22ecb364f9e",
        addresses=(0, 1, 2, 3, 682, 1024, 2047),
        sign_bits=(
            0x0000000000000000,
            0x1086C5ABC9DAEEFF,
            0x30CF0FFC0F3F3300,
            0x2049CA57C6E5DDFF,
            0x03CFCCC030CF0FFC,
            0x5555555555555555,
            0x572FDD297AD05F32,
        ),
    ),
    (8, "kerdock"): GoldenCodebook(
        capacity=32768,
        dim=256,
        sha256="e47e935ba50f6227e0a218ba2ff8fea48eb0d56c15efdd17f2ad88bcb210b643",
        addresses=(0, 1, 2, 3, 10922, 16384, 32767),
        sign_bits=(
            0x0000000000000000000000000000000000000000000000000000000000000000,
            0x20008C428F49446E8339CADDC54DA7AE00B4B65782EFD79B09958FDA29BE2B20,
            0x3000C0C3C0CFCCF3C00F0F330FC3FCF300CCCFFCC3303CFC0FFFC03F3FC33C30,
            0x10004C814F86889D4336C5EECA8E5B5D007879AB41DFEB67066A4FE5167D1710,
            0x3FF00FCFF0CF0C3000C0C3C0CFCCF3C00F0F330FC3FCF300CCCFFCC3303CFC0F,
            0x5555555555555555555555555555555555555555555555555555555555555555,
            0x7F0AA32FC9EF614A9AE91B16E3DD08D5A35F776C1B01076622EF02E78A72FDAF,
        ),
    ),
    (6, "dg1"): GoldenCodebook(
        capacity=65536,
        dim=64,
        sha256="06d1f0bee3a094c2c6374a29bc7f1e9e1ca98060c53efd2a578fbabd6bdde82b",
        addresses=(0, 1, 2, 3, 21845, 32768, 65535),
        sign_bits=(
            0x0000000000000000,
            0x1086C5ABC9DAEEFF,
            0x30CF0FFC0F3F3300,
            0x2049CA57C6E5DDFF,
            0x681C1D267A105CF1,
            0x0FC0CCF300F0C3FF,
            0x541C112AB9DFA3CE,
        ),
    ),
}


def _sign_bits(vector: torch.Tensor) -> int:
    bits = 0
    for value in vector.tolist():
        bits = (bits << 1) | int(value < 0)
    return bits


@pytest.mark.parametrize(("m", "code_type"), sorted(GOLDEN))
def test_codebook_matches_golden(m: int, code_type: str):
    golden = GOLDEN[(m, code_type)]
    generator = KerdockDGCodeGenerator(m=m, code_type=code_type)

    assert generator.capacity == golden.capacity
    assert golden.dim == generator.N

    codebook = generator.precompute_codebook(dtype=torch.float32).contiguous()
    assert codebook.shape == (golden.capacity, golden.dim)

    for address, expected in zip(golden.addresses, golden.sign_bits):
        assert _sign_bits(codebook[address]) == expected, (
            f"sign pattern changed at address {address}"
        )

    digest = hashlib.sha256(codebook.numpy().tobytes()).hexdigest()
    assert digest == golden.sha256, (
        "Kerdock/DG codebook bytes changed. Composed Granite Switch checkpoints "
        "encode adapter addresses against this exact construction, so a change "
        "here silently routes tokens to the wrong adapter."
    )


@pytest.mark.parametrize(("m", "code_type"), sorted(GOLDEN))
def test_codebook_is_unit_norm_and_antipodal_free(m: int, code_type: str):
    generator = KerdockDGCodeGenerator(m=m, code_type=code_type)
    codebook = generator.precompute_codebook(dtype=torch.float32)

    norms = torch.linalg.norm(codebook, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms))

    # Coherence must stay strictly below 1: a pair of antipodal addresses is
    # indistinguishable after retrieval. Checking the full Gram matrix is
    # quadratic, so bound it on a deterministic slice.
    head = codebook[:512]
    gram = head @ head.t()
    gram.fill_diagonal_(0.0)
    assert gram.abs().max().item() <= generator.coherence + 1e-5


def test_oversized_codebook_is_rejected():
    # DG(8,1) addresses 4.2M codes of dimension 256, i.e. 4 GiB in float32.
    generator = KerdockDGCodeGenerator(m=8, code_type="dg1")
    assert generator.capacity == 2 ** (3 * 8 - 2)
    with pytest.raises(ValueError, match="too large to precompute"):
        generator.precompute_codebook()


@pytest.mark.parametrize("code_type", ["kerdock", "dg1"])
def test_invalid_m_is_rejected(code_type: str):
    with pytest.raises(ValueError, match="m must be 6 or 8"):
        KerdockDGCodeGenerator(m=7, code_type=code_type)  # type: ignore[arg-type]


def test_recover_count_from_signal_is_exact_over_capacity():
    capacity = 2048
    counts = torch.arange(capacity, dtype=torch.float32)
    signal = 1.0 / (1.0 + counts)
    torch.testing.assert_close(
        recover_count_from_signal(signal, capacity=capacity),
        counts.long(),
    )


def test_recover_count_from_signal_upcasts_bfloat16():
    # The signal is read out of a paged KV cache, so it arrives in the cache
    # dtype. bfloat16 has a ULP of 0.5 over [64, 128), which aliases counts
    # above 188; below that the float32 inversion must still be exact.
    counts = torch.arange(189, dtype=torch.float32)
    signal = (1.0 / (1.0 + counts)).bfloat16()
    torch.testing.assert_close(
        recover_count_from_signal(signal, capacity=2048),
        counts.long(),
    )


def test_recover_count_from_signal_clamps_to_capacity():
    # A signal smaller than 1/capacity means more control tokens than the
    # codebook can address; the count must saturate rather than index
    # out of bounds.
    signal = torch.tensor([1.0 / 100000.0, 1.0], dtype=torch.float32)
    assert recover_count_from_signal(signal, capacity=2048).tolist() == [2047, 0]
