# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch


def _install_fake_amdsmi(
    monkeypatch: pytest.MonkeyPatch,
    target_graphics_version: str,
) -> None:
    fake_amdsmi = types.ModuleType("amdsmi")

    class AmdSmiException(Exception):
        pass

    fake_amdsmi.AmdSmiException = AmdSmiException
    fake_amdsmi.amdsmi_init = lambda: None
    fake_amdsmi.amdsmi_shut_down = lambda: None
    fake_amdsmi.amdsmi_get_processor_handles = lambda: ["gpu0"]
    fake_amdsmi.amdsmi_get_gpu_asic_info = lambda handle: {
        "target_graphics_version": target_graphics_version
    }
    fake_amdsmi.amdsmi_get_gpu_device_uuid = lambda handle: "uuid"
    fake_amdsmi.amdsmi_topo_get_link_type = lambda src, dst: {"hops": 1, "type": 2}
    monkeypatch.setitem(sys.modules, "amdsmi", fake_amdsmi)


def _load_rocm_platform(
    monkeypatch: pytest.MonkeyPatch,
    target_graphics_version: str,
):
    _install_fake_amdsmi(monkeypatch, target_graphics_version)
    sys.modules.pop("vllm.platforms.rocm", None)
    rocm = importlib.import_module("vllm.platforms.rocm")
    rocm.use_rocm_custom_paged_attention.cache_clear()
    return rocm


@pytest.mark.parametrize(
    ("target_graphics_version", "head_size", "gqa_ratio", "expected"),
    [
        # gfx950 + gqa=2 currently triggers the inaccurate mfma4 path tracked in
        # issue #35569.
        ("gfx950", 128, 2, False),
        # gfx942 + gqa=4 currently triggers the crashing mfma4 path tracked in
        # issue #36180.
        ("gfx942", 64, 4, False),
        # mfma16-backed shapes should keep using the native kernel.
        ("gfx950", 128, 5, True),
    ],
)
def test_use_rocm_custom_paged_attention_avoids_bad_mfma4_shapes(
    monkeypatch: pytest.MonkeyPatch,
    target_graphics_version: str,
    head_size: int,
    gqa_ratio: int,
    expected: bool,
) -> None:
    rocm = _load_rocm_platform(monkeypatch, target_graphics_version)

    assert (
        rocm.use_rocm_custom_paged_attention(
            qtype=torch.bfloat16,
            head_size=head_size,
            block_size=16,
            gqa_ratio=gqa_ratio,
            max_seq_len=8192,
            sliding_window=0,
            kv_cache_dtype="auto",
        )
        is expected
    )
