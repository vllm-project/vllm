# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""amd_kernels KDA prefill hook (akr_kda_prefill.try_prefill) vs the stock prefill of
KimiK3DeltaAttention._forward, with the state indices the GDN builder passes: the column
view block_table[:, 0] of a (requests, 1 + speculative blocks) table. Every sequence
must read and write its own state slot (single GPU)."""

import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("amd_kernels")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "amd_prefill"))
import check_kda_prefill as C  # noqa: E402

CASES = [
    ("two_fresh", 0, [300, 500], [0, 0]),
    ("three_ctx", 0, [700, 300, 1500], [5000, 0, 64000]),
    ("five_ragged", 0, [300, 900, 1200, 450, 1500], [0, 0, 0, 0, 0]),
    ("mixed_decodes", 4, [400, 700, 1100], [0, 2000, 0]),
]


@pytest.fixture(scope="module")
def env():
    import vllm.models.kimi_k3.amd.kda as kda
    from vllm.config import VllmConfig, set_current_vllm_config

    saved = kda.get_forward_context, os.environ.get("VLLM_KIMI_AKR_KDA_PREFILL")

    def patch(f):
        kda.get_forward_context = f

    with set_current_vllm_config(VllmConfig()):
        yield C.make_layer(), patch
    kda.get_forward_context = saved[0]
    if saved[1] is None:
        os.environ.pop("VLLM_KIMI_AKR_KDA_PREFILL", None)
    else:
        os.environ["VLLM_KIMI_AKR_KDA_PREFILL"] = saved[1]


@pytest.mark.parametrize("name,nd,plens,ctxs", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_strided_state_indices(env, name, nd, plens, ctxs):
    layer, patch = env
    b = C.make_batch(nd, plens, ctxs)
    slots = b["m"].non_spec_state_indices_tensor
    assert not slots.is_contiguous()
    o0, r0, c0, _ = C.run(layer, b, False, patch)
    o1, r1, c1, called = C.run(layer, b, True, patch)
    assert called == [True]
    assert C.rel(o1[0], o0[0]) < 1e-2
    for s in slots.tolist():
        assert C.rel(r1[s], r0[s]) < 1e-2, f"state slot {s}"
        assert torch.equal(c1[s], c0[s]), f"conv slot {s}"
    others = torch.ones(r0.shape[0], dtype=torch.bool, device=r0.device)
    others[slots.long()] = False
    others[0] = False  # the null block (the kernel entry's first-call warm-up)
    assert torch.equal(r1[others], b["rec"][others])
