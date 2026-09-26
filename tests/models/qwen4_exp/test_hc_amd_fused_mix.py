# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused HyperConnection low-rank mix on ROCm.

The fused kernels replace four launches with two and must agree with the path
they replace. Each test drives the fused kernel and the unfused reference
composed from the ops already in the tree, so a regression shows up as a
numerical difference rather than as a shape error.
"""

import pytest
import torch

from vllm.models.qwen4_exp.amd.ops.hc import (
    HC_FUSED_MIX_MAX_TOKENS,
    hc_down_silu,
    hc_gate_mix,
    hc_silu,
    hc_up_gate_mix,
    supports_fused_low_rank_mix,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON,
    reason="AMD HC kernels require ROCm and Triton",
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
LORA_RANK = 320
# lora_rank + hc_count rounded up to the 16-row alignment the merged
# projection is padded to.
DOWN_OUT = 336


def _operands(num_tokens: int, dtype=torch.bfloat16):
    torch.manual_seed(0)
    xn = torch.randn(num_tokens, HYPER_HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.1
    w_down = torch.randn(DOWN_OUT, HYPER_HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    w_up = torch.randn(HYPER_HIDDEN_SIZE, LORA_RANK, dtype=dtype, device="cuda") * 0.02
    return xn, w_down, w_up


@pytest.mark.parametrize("num_tokens", [1, 2, 4, HC_FUSED_MIX_MAX_TOKENS])
def test_down_silu_matches_unfused(num_tokens: int) -> None:
    xn, w_down, _ = _operands(num_tokens)

    actual = hc_down_silu(xn, w_down, LORA_RANK, HC)

    down = torch.nn.functional.linear(xn, w_down)
    expected = down.clone()
    expected[:, :LORA_RANK] = hc_silu(down[:, :LORA_RANK].contiguous(), HC)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, HC_FUSED_MIX_MAX_TOKENS])
def test_up_gate_mix_matches_unfused(num_tokens: int) -> None:
    xn, _, w_up = _operands(num_tokens)
    lora = torch.randn(num_tokens, LORA_RANK, dtype=xn.dtype, device="cuda") * 0.1

    actual = hc_up_gate_mix(lora, w_up, xn, HC)

    gate = torch.nn.functional.linear(lora, w_up)
    expected = hc_gate_mix(xn, gate, HC)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_injection_columns_pass_through_unactivated() -> None:
    """The merged projection carries injection logits the SiLU must not touch.

    Activating them would be silent: they feed the next combine rather than
    this block's output, so the shapes stay right and only the model's
    behaviour changes.
    """
    xn, w_down, _ = _operands(4)

    actual = hc_down_silu(xn, w_down, LORA_RANK, HC)
    expected = torch.nn.functional.linear(xn, w_down)

    torch.testing.assert_close(
        actual[:, LORA_RANK:], expected[:, LORA_RANK:], atol=2e-2, rtol=2e-2
    )
    # ...and the low-rank half must *not* match the raw projection, or the
    # test above would pass with the activation missing entirely.
    assert not torch.allclose(
        actual[:, :LORA_RANK], expected[:, :LORA_RANK], atol=2e-2, rtol=2e-2
    )


def test_fused_path_is_gated_to_decode_widths() -> None:
    """Above the skinny-GEMM width the fused kernels must not be selected.

    They do not tile M, so they degrade quickly, and the unfused path stops
    using `wvSplitK` there anyway. Prefill has to fall back.
    """
    _, w_down, w_up = _operands(1)

    for num_tokens in (1, HC_FUSED_MIX_MAX_TOKENS):
        xn = torch.empty(
            num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        assert supports_fused_low_rank_mix(xn, w_down, w_up)

    for num_tokens in (HC_FUSED_MIX_MAX_TOKENS + 1, 2048):
        xn = torch.empty(
            num_tokens, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        assert not supports_fused_low_rank_mix(xn, w_down, w_up)

    # A transposed weight keeps its shape but loses the unit inner stride the
    # kernels index with, so it must fall back too.
    xn = torch.empty(4, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    assert not supports_fused_low_rank_mix(xn, w_down.t().contiguous().t(), w_up)


def test_module_fused_and_unfused_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the real module both ways.

    The kernels are covered above; what this covers is the wiring around them,
    which is where the quiet mistakes live -- reading the wrong weight
    attribute, or slicing the injection logits at the wrong offset now that
    the merged projection is consumed directly rather than through
    ``Tensor.split``.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.qwen4_exp.amd import hyperconnection as hc_module
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig

    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        hc_lowrank=LORA_RANK,
    )
    torch.manual_seed(0)
    # The Linear modules on the unfused side are CustomOps: a current config
    # has to be set to build the parallel groups, to instantiate them, and to
    # run them.
    with set_current_vllm_config(VllmConfig()):
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29513",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)

        module = hc_module.GatedResidual(config, use_combine=True).cuda()
        for param in module.parameters():
            torch.nn.init.normal_(param, std=0.02)

        xn = (
            torch.randn(4, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1
        )

        from vllm.models.qwen4_exp.amd.ops import hc as hc_ops

        assert supports_fused_low_rank_mix(
            xn,
            module.input_mix_weight_down_block_inject.weight,
            module.input_mix_weight_up.weight,
        )
        fused_out, fused_inj = module._low_rank_mix(xn)

        # The predicate lives in the ops now, so that is where it has to be
        # stubbed to exercise the fallback.
        monkeypatch.setattr(hc_ops, "supports_fused_low_rank_mix", lambda *a: False)
        unfused_out, unfused_inj = module._low_rank_mix(xn)

    torch.testing.assert_close(fused_out, unfused_out, atol=2e-2, rtol=2e-2)
    assert fused_inj.shape == unfused_inj.shape == (4, HC)
    torch.testing.assert_close(fused_inj, unfused_inj, atol=2e-2, rtol=2e-2)


def test_fused_path_is_reached_without_recompiling() -> None:
    """The width test has to live in the op, not in the compiled caller.

    vLLM compiles this model ahead of time as a single graph spanning the
    whole token range. A Python branch on the batch width in the caller is
    therefore resolved once -- against the memory profile run, which is
    thousands of tokens wide -- and frozen, and the recompilation that would
    otherwise rescue such a branch never happens. An earlier revision of this
    code had the branch in the caller and shipped a fused path that no served
    request ever reached, while every eager test kept passing.

    Encode that property directly: warm the graph at prefill widths, forbid
    recompilation, then call at a decode width and require the fused kernels
    to run anyway.
    """
    if not hasattr(torch.compiler, "set_stance"):
        pytest.skip("torch.compiler.set_stance is unavailable")

    from vllm.models.qwen4_exp.amd.ops import hc as hc_ops

    _, w_down, w_up = _operands(1)

    def mix(xn: torch.Tensor) -> torch.Tensor:
        down = hc_ops.hc_down_silu(xn, w_down, LORA_RANK, HC)
        return hc_ops.hc_up_gate_mix(down[:, :LORA_RANK], w_up, xn, HC)

    compiled = torch.compile(mix, fullgraph=True, dynamic=True)
    # Two widths so the batch dimension really is traced as dynamic rather
    # than specialised to the first one.
    for width in (64, 128):
        compiled(
            torch.randn(
                width, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
            )
            * 0.1
        )

    xn = torch.randn(4, HYPER_HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1
    before = hc_ops.fused_call_count
    with torch.compiler.set_stance("fail_on_recompile"):
        out = compiled(xn)

    assert hc_ops.fused_call_count > before, (
        "the compiled graph never reached the fused kernels; the width test "
        "has leaked back into the caller"
    )
    torch.testing.assert_close(out, mix(xn), atol=2e-2, rtol=2e-2)


def test_full_chain_matches_reference() -> None:
    """Both fused kernels together, against a plain torch reference."""
    xn, w_down, w_up = _operands(4)

    down = hc_down_silu(xn, w_down, LORA_RANK, HC)
    actual = hc_up_gate_mix(down[:, :LORA_RANK], w_up, xn, HC)

    lora = torch.nn.functional.linear(xn.float(), w_down.float())[:, :LORA_RANK] / HC
    silu = lora * torch.sigmoid(lora)
    gate = torch.nn.functional.linear(silu, w_up.float())
    expected = (
        torch.sigmoid(gate).unflatten(-1, (HC, HIDDEN_SIZE))
        * xn.float().unflatten(-1, (HC, HIDDEN_SIZE))
    ).mean(-2)

    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=3e-2)
