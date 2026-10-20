# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adversarial one-GPU stress of the same-GPU hand-off protocol across XCDs
(kernel: ``_xcd_stress.py``): every hand-off between CTAs placed on different
XCDs must deliver the producer's current payload, with no wait timing out."""

import os

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 only", allow_module_level=True)

from tests.models.minimax_m3.amd_specialized_decode import (  # noqa: E402
    _xcd_stress as xs,
)
from vllm.models.minimax_m3.amd.specialized_decode.mono.runner import (  # noqa: E402
    _writable_flydsl_cache,
)

LAUNCHES = int(os.environ.get("MINIMAX_M3_XCD_STRESS_LAUNCHES", "200"))


def test_handoffs_between_xcds():
    _writable_flydsl_cache()
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    assert cus >= xs.BLOCKS, "every CTA must be resident"
    err, out = xs.run(LAUNCHES)
    assert err[0].item() == 0, err.tolist()
    bad = out[:, 2]
    assert bad.sum().item() == 0, out[bad.nonzero().flatten()][:8].tolist()

    xcc = out[:, 0]
    assert sorted(set(xcc.tolist())) == list(range(8))
    # every round's producer / consumer pair on two XCDs
    blocks = torch.arange(xs.BLOCKS)
    for k in range(1, 128, 2):
        assert (xcc != xcc[(blocks + xs.BLOCKS - k) % xs.BLOCKS]).all(), k


@pytest.mark.parametrize("stores, loads", [(False, True), (True, False)])
def test_plain_payload_accesses_are_caught(stores, loads):
    """The stress has teeth: with either side of the payload a plain cached
    access, consumers read stale data."""
    _writable_flydsl_cache()
    err, out = xs.run(4, stores, loads, stop_on_error=False)
    assert err[0].item() == 0, err.tolist()
    assert (out[:, 2] > 0).any()
