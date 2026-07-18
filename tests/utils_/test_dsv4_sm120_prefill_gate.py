# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSv4 SM120 packed-prefill backend-active gate.

``is_dsv4_sm120_fi_prefill_active()`` is the signal the SHARED
``DeepseekSparseSWAMetadataBuilder`` uses to decide whether to launch the
prefill-SWA index kernel. Since ``VLLM_DEEPSEEK_V4_FLASHINFER_SM120_PREFILL`` now
defaults ON, this gate is what keeps the default FlashMLA/Triton path from
launching that kernel (which faults). It must be True ONLY when all three of
{DECODE opted in, SM12x device, FI SM120 kernel present} hold.
"""
from unittest.mock import patch

import pytest

from vllm.utils import flashinfer


@pytest.fixture(autouse=True)
def _clear_cache():
    flashinfer.is_dsv4_sm120_fi_prefill_active.cache_clear()
    yield
    flashinfer.is_dsv4_sm120_fi_prefill_active.cache_clear()


def _run(monkeypatch, decode: bool, family120: bool, kernel: bool) -> bool:
    # The DECODE env resolves through os.environ via the envs.py getter lambda.
    monkeypatch.setenv(
        "VLLM_DEEPSEEK_V4_FLASHINFER_SM120_DECODE", "1" if decode else "0"
    )
    with (
        patch(
            "vllm.platforms.current_platform.is_device_capability_family",
            return_value=family120,
        ),
        patch.object(
            flashinfer,
            "has_flashinfer_trtllm_sparse_mla_dsv4",
            return_value=kernel,
        ),
    ):
        flashinfer.is_dsv4_sm120_fi_prefill_active.cache_clear()
        return flashinfer.is_dsv4_sm120_fi_prefill_active()


def test_all_three_true_is_active(monkeypatch):
    assert _run(monkeypatch, decode=True, family120=True, kernel=True) is True


@pytest.mark.parametrize(
    "decode,family120,kernel",
    [
        (False, True, True),   # SM120 FI decode backend not opted in (the default)
        (True, False, True),   # not SM12x (e.g. Hopper)
        (True, True, False),   # FI SM120 kernel absent (flashinfer < 0.6.13)
        (False, False, False),
    ],
)
def test_any_condition_false_is_inactive(monkeypatch, decode, family120, kernel):
    # Safety-critical: DECODE off (the default) -> inactive, so the shared SWA
    # builder must NOT launch the prefill-SWA kernel on the default FlashMLA path.
    assert _run(monkeypatch, decode=decode, family120=family120, kernel=kernel) is False
