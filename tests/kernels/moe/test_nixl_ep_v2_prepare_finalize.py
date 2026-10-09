# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the nixl_ep_v2 prepare-finalize: dispatch and combine round
trips in each receive layout, and the expand knob."""

import types

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import get_ep_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_nixl_ep_v2
from vllm.utils.math_utils import next_power_of_2


def _nixl_ep_v2_layout_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    use_cudagraph: bool,
    use_fp8: bool,
    skewed: bool,
    expand: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep_v2 import (
        NixlEPV2PrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    world, rank, device = pgi.world_size, pgi.rank, pgi.device
    num_local_experts, hidden, topk = 4, 256, 2
    num_tokens = [5, 3]  # per rank; the bucket is the largest's power of two
    # Every rank draws the whole routing, so it knows what it receives.
    # Skewed: every token picks rank 0's experts, so rank 1 receives nothing.
    gen = torch.Generator().manual_seed(0)
    num_choices = num_local_experts * (1 if skewed else world)
    all_ids = [
        torch.stack([torch.randperm(num_choices, generator=gen, device="cpu")[:topk]
                     for _ in range(n)])
        for n in num_tokens
    ]  # fmt: skip
    owner = torch.cat(all_ids).long() // num_local_experts
    num_recv_slots = int((owner == rank).sum())
    num_recv_tokens = int((owner == rank).any(dim=1).sum())

    mgr = get_ep_group().device_communicator.all2all_manager
    buffer = mgr.get_handle(
        dict(
            num_max_tokens_per_rank=16,
            hidden=hidden,
            num_topk=topk,
            use_fp8_dispatch=use_fp8,
            num_local_experts=num_local_experts,
        )
    )
    pf = NixlEPV2PrepareAndFinalize(
        buffer,
        num_dispatchers=world,
        rank_expert_offset=rank * num_local_experts,
        num_sms=mgr.num_sms,
        use_cudagraph=use_cudagraph,
        expand=expand,
    )
    quant_config = (
        FusedMoEQuantConfig.make(current_platform.fp8_dtype(), block_shape=[128, 128])
        if use_fp8
        else FusedMoEQuantConfig.make()
    )
    a1 = torch.randn(num_tokens[rank], hidden, dtype=torch.bfloat16, device=device)
    topk_ids = all_ids[rank].long().to(device)
    topk_weights = torch.rand(num_tokens[rank], topk, device=device)
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens[rank],
        num_tokens_across_dp=torch.tensor(num_tokens, dtype=torch.int),
    ):
        x, x_scale, meta, ids, weights = pf.prepare(
            a1,
            topk_weights,
            topk_ids,
            num_local_experts * world,
            None,
            False,
            quant_config,
        )
        x = x.float()
        if use_fp8:  # dequantize the 128-wide blocks
            x = x.view(len(x), hidden // 128, 128) * x_scale.unsqueeze(-1)
            x = x.view(len(x), hidden)
        local = (ids >= rank * num_local_experts) & (
            ids < (rank + 1) * num_local_experts
        )
        if use_cudagraph and expand:
            # Expanded for the worst case: the padded rows times
            # min(topk, local experts), grouped by expert.
            assert len(x) == next_power_of_2(max(num_tokens)) * world * min(
                topk, num_local_experts
            )
            assert meta.expert_num_tokens is None
            assert int(meta.psum_recv_per_rank[-1]) == num_recv_slots
            assert (ids[num_recv_slots:] == -1).all()
            x[num_recv_slots:] = 0  # uninitialized padding rows
            assert local[:num_recv_slots].all()
            assert (ids[:num_recv_slots].diff(dim=0) >= 0).all()
        elif use_cudagraph:
            # Padded: one row per token, the bucket for the maximum world.
            assert len(x) == next_power_of_2(max(num_tokens)) * world
            assert meta.expert_num_tokens is None
            assert int(meta.psum_recv_per_rank[-1]) == num_recv_tokens
            assert (ids[num_recv_tokens:] == -1).all()
            assert int(local[:num_recv_tokens].sum()) == num_recv_slots
            assert ((ids == -1) | local).all()
        else:
            # Exact: one row per (token, local expert), grouped by expert.
            assert len(x) == num_recv_slots == int(meta.expert_num_tokens.sum())
            assert len(meta.expert_num_tokens) == num_local_experts
            assert local.all() and (ids.diff(dim=0) >= 0).all()
        # Experts that return each row times its local weights.
        out = (x * (weights * local).sum(dim=1, keepdim=True)).to(torch.bfloat16)
        output = torch.empty_like(a1)
        pf.finalize_async(output, out, weights, ids, False, TopKWeightAndReduceNoOP())()
    # The dispatch committed the manager's active SM count.
    assert buffer.get_topology()[1] == mgr.max_sms_used()

    if use_fp8:
        q, scale = moe_kernel_quantize_input(
            a1, None, current_platform.fp8_dtype(), False, [128, 128], False
        )
        a1 = q.float().view(len(q), hidden // 128, 128) * scale.unsqueeze(-1)
        a1 = a1.view(len(q), hidden)
    expected = (a1.float() * topk_weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("skewed", [False, True])
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("use_cudagraph", [False, True])
def test_nixl_ep_v2_prepare_finalize(
    use_cudagraph: bool, use_fp8: bool, skewed: bool
) -> None:
    """Dispatch and combine round-trip tokens in both receive layouts.

    Args:
        use_cudagraph: Take the padded layout instead of the exact one.
        use_fp8: Dispatch block-FP8 activations instead of BF16.
        skewed: Route every token to rank 0, so rank 1 receives no rows.

    """
    _launch_nixl_ep_v2_layout(None, use_cudagraph, use_fp8, skewed, expand=False)


@pytest.mark.parametrize("skewed", [False, True])
@pytest.mark.parametrize("use_fp8", [False, True])
def test_nixl_ep_v2_prepare_finalize_expand(use_fp8: bool, skewed: bool) -> None:
    """Dispatch and combine round-trip tokens in the worst-case expanded
    layout that VLLM_NIXL_EP_V2_EXPAND selects under CUDA graphs.

    Args:
        use_fp8: As in test_nixl_ep_v2_prepare_finalize.
        skewed: Route every token to rank 0, so rank 1 receives no rows.

    """
    _launch_nixl_ep_v2_layout(None, True, use_fp8, skewed, expand=True)


def test_nixl_ep_v2_prepare_finalize_knobs() -> None:
    """The VLLM_NIXL_EP_V2_* buffer and SM-count knobs reach the kernels and
    still round-trip tokens."""
    env = {
        "VLLM_NIXL_EP_V2_NUM_SMS": "16",  # of the 26-SM launch grid
        "VLLM_NIXL_EP_V2_PREFER_OVERLAP": "0",
        "VLLM_NIXL_EP_V2_ALLOW_MULTIPLE_REDUCTION": "0",
    }
    _launch_nixl_ep_v2_layout(env, False, False, False, expand=False)


def _launch_nixl_ep_v2_layout(
    env: dict[str, str] | None,
    use_cudagraph: bool,
    use_fp8: bool,
    skewed: bool,
    expand: bool,
) -> None:
    if not has_nixl_ep_v2():
        pytest.skip("Requires NIXL EP v2 (nixl.runtime.ep)")
    if current_platform.device_count() < 2:
        pytest.skip("Requires 2 GPUs")

    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            all2all_backend="nixl_ep_v2",
        ),
    )
    parallel_launch_with_config(
        2,
        _nixl_ep_v2_layout_worker,
        vllm_config,
        env,
        use_cudagraph,
        use_fp8,
        skewed,
        expand,
    )


@pytest.mark.parametrize("env", [None, "1"], ids=["unset", "1"])
def test_nixl_ep_v2_expand_knob(
    env: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under CUDA graphs, VLLM_NIXL_EP_V2_EXPAND switches the dispatch to the
    worst-case expanded layout with no CPU sync, with IDs from the device.

    Args:
        env: VLLM_NIXL_EP_V2_EXPAND, or None to leave it unset.

    """
    if not has_nixl_ep_v2():
        pytest.skip("Requires NIXL EP v2 (nixl.runtime.ep)")
    from tests.kernels.moe.utils import make_ep_prepare_finalize_with_fake_buffer

    class _Buffer:
        num_launch_sms = 26  # the fake manager's num_sms too
        num_max_tokens_per_rank = 16

        def dispatch(self, x, **kwargs):
            self.kwargs = kwargs
            # 8 worst-case rows; local experts 0 and 2 got 2 and 3 of them.
            handle = types.SimpleNamespace(
                recv_count_is_exact=False,
                psum_num_recv_tokens_per_expert=torch.tensor(
                    [2, 2, 5, 5], dtype=torch.int32
                ),
            )
            return x.new_zeros(8, x.size(1)), None, torch.ones(8), handle, None

    monkeypatch.delenv("VLLM_NIXL_EP_V2_EXPAND", raising=False)
    if env is not None:
        monkeypatch.setenv("VLLM_NIXL_EP_V2_EXPAND", env)
    buffer = _Buffer()
    pf = make_ep_prepare_finalize_with_fake_buffer("nixl_ep_v2", buffer, monkeypatch)
    a1 = torch.randn(3, 256, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 4], [1, 2], [5, 6]])
    with set_forward_context(None, VllmConfig()):
        receiver = pf.prepare_async(
            a1, torch.rand(3, 2), topk_ids, 8, None, False, FusedMoEQuantConfig.make()
        )
    # No CPU sync and the power-of-two bucket either way.
    assert buffer.kwargs["do_expand"] is (env is not None)
    assert buffer.kwargs["do_cpu_sync"] is False
    assert buffer.kwargs["num_max_tokens_per_rank"] == 4
    assert buffer.kwargs["num_sms"] == 26
    if env is None:
        return  # the padded receiver needs a GPU
    _, _, meta, ids, weights = receiver()
    assert ids.flatten().tolist() == [0, 0, 2, 2, 2, -1, -1, -1]
    assert meta.expert_num_tokens is None
    assert int(meta.psum_recv_per_rank[-1]) == 5
    assert weights.shape == (8, 1)


@pytest.mark.parametrize(
    "flag,quant,defer,swizzled,wire",
    [
        (False, "nvfp4", False, False, "bf16"),
        (True, "nvfp4", False, False, "nvfp4"),
        (True, "nvfp4", False, True, "nvfp4"),
        (True, "nvfp4", True, False, "bf16"),
        (True, None, False, False, "bf16"),
    ],
    ids=["off", "on", "on-swizzled", "on-deferred", "on-bf16-layer"],
)
def test_nixl_ep_v2_nvfp4_dispatch_knob(
    flag: bool,
    quant: str | None,
    defer: bool,
    swizzled: bool,
    wire: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VLLM_NIXL_EP_V2_NVFP4_DISPATCH quantizes NVFP4 layers' input before
    the dispatch and sends (codes, row-major scales); the receiver does not
    requantize, and swizzles the scales only when the experts ask. Deferred
    or non-NVFP4 layers dispatch BF16 as without it.

    Args:
        flag: Set VLLM_NIXL_EP_V2_NVFP4_DISPATCH.
        quant: The layer's quant_dtype: "nvfp4" or None for BF16.
        defer: The experts quantize their own input.
        swizzled: The experts want swizzled scales.
        wire: The expected dispatch payload.

    """
    if not has_nixl_ep_v2():
        pytest.skip("Requires NIXL EP v2 (nixl.runtime.ep)")
    import vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep_v2 as pf_mod
    from tests.kernels.moe.utils import make_ep_prepare_finalize_with_fake_buffer

    num_tokens, hidden = 3, 256
    quant_calls: list[dict] = []
    swizzle_calls: list[torch.Tensor] = []

    def fake_quantize(a, a_scale, quant_dtype, per_act_token_quant, **kwargs):
        # scaled_fp4_quant's unswizzled shapes; the real op needs SM100.
        quant_calls.append(dict(a_scale=a_scale, quant_dtype=quant_dtype, **kwargs))
        n = len(a)
        return (
            torch.zeros(n, hidden // 2, dtype=torch.uint8),
            torch.zeros(n, hidden // 16, dtype=torch.uint8).view(torch.float8_e4m3fn),
        )

    def fake_interleave(scale):
        swizzle_calls.append(scale)
        return torch.ones(128, scale.size(1), dtype=torch.uint8)

    class _Buffer:
        num_launch_sms = 26
        num_max_tokens_per_rank = 16

        def dispatch(self, x, **kwargs):
            self.x = x
            # Exact layout: 4 rows, 1 + 3 for local experts 0 and 2.
            handle = types.SimpleNamespace(
                recv_count_is_exact=True, num_recv_tokens_per_expert_list=[1, 0, 3, 0]
            )
            if isinstance(x, tuple):
                recv_x = (
                    x[0].new_zeros(4, x[0].size(1)),
                    x[1].new_zeros(4, x[1].size(1)),
                )
            else:
                recv_x = x.new_zeros(4, x.size(1))
            return recv_x, None, torch.ones(4), handle, None

    monkeypatch.setattr(pf_mod, "moe_kernel_quantize_input", fake_quantize)
    monkeypatch.setattr(pf_mod, "nvfp4_block_scale_interleave", fake_interleave)
    monkeypatch.delenv("VLLM_NIXL_EP_V2_NVFP4_DISPATCH", raising=False)
    if flag:
        monkeypatch.setenv("VLLM_NIXL_EP_V2_NVFP4_DISPATCH", "1")
    buffer = _Buffer()
    pf = make_ep_prepare_finalize_with_fake_buffer("nixl_ep_v2", buffer, monkeypatch)
    assert pf.nvfp4_dispatch is flag
    gscale = torch.tensor([2.0, 2.0])  # the oracle repeats one value per expert
    quant_config = (
        FusedMoEQuantConfig.make("nvfp4", a1_gscale=gscale, is_scale_swizzled=swizzled)
        if quant == "nvfp4"
        else FusedMoEQuantConfig.make()
    )
    a1 = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    with set_forward_context(None, VllmConfig()):
        receiver = pf.prepare_async(
            a1,
            torch.rand(num_tokens, 2),
            torch.tensor([[0, 4], [1, 2], [5, 6]]),
            8,
            None,
            False,
            quant_config,
            defer_input_quant=defer,
        )
    x, x_scale, meta, ids, _ = receiver()
    assert ids.flatten().tolist() == [0, 2, 2, 2]

    if wire == "bf16":
        assert buffer.x is a1
        # Without the flag, the NVFP4 layer still quantizes after receive.
        assert len(quant_calls) == int(quant == "nvfp4" and not defer)
        return
    codes, scales = buffer.x
    assert codes.dtype == torch.uint8 and codes.shape == (num_tokens, hidden // 2)
    assert scales.dtype == torch.float8_e4m3fn
    assert scales.shape == (num_tokens, hidden // 16)
    # Quantized once, before the dispatch, row-major with the layer's scale.
    [call] = quant_calls
    assert call["a_scale"] is gscale
    assert call["quant_dtype"] == "nvfp4" and call["is_scale_swizzled"] is False
    assert x.dtype == torch.uint8 and x.shape == (4, hidden // 2)
    if swizzled:
        [swizzle_in] = swizzle_calls
        assert swizzle_in.dtype == torch.uint8 and swizzle_in.shape == (4, hidden // 16)
        assert x_scale.shape == (128, hidden // 16)
    else:
        assert not swizzle_calls
        assert x_scale.dtype == torch.float8_e4m3fn
        assert x_scale.shape == (4, hidden // 16)
