# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockScaleSplitKZeroInitFusionPass``.

The pass rewrites a chain of
    (functional producer) -> (functional blockscale GEMM)
into
    torch.empty -> auto_functionalized(producer_with_zero_init, gemm_out_zero_init=Y)
                -> auto_functionalized(gemm_splitk, output=Y, split_k=k, y_is_zeroed=True)

The tests below exercise this rewrite on toy modules covering each producer
listed in the default registry (P1 per-token-quant, P2 Gemma RMSNorm, P3
gated RMSNorm). All tests are gated behind ``is_aiter_found_and_supported``
because they call ROCm AITER ops at trace-time.
"""

from __future__ import annotations

import pytest
import torch

import vllm.config
from tests.compile.backend import TestBackend
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)


# ---------------------------------------------------------------------------
# Always-on smoke tests of the registry module (no ROCm required)
# ---------------------------------------------------------------------------


def test_registries_module_importable():
    """The fusion pass module must be importable; on non-ROCm boxes (or with
    AITER disabled) ``build_default_registries`` returns empty lists and the
    pass becomes a no-op. On ROCm+AITER it returns populated lists."""
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        BlockScaleSplitKZeroInitFusionPass,
        GemmSpec,
        ProducerSpec,
        build_default_registries,
    )

    producers, gemms = build_default_registries()
    assert isinstance(producers, list)
    assert isinstance(gemms, list)
    assert all(isinstance(p, ProducerSpec) for p in producers)
    assert all(isinstance(g, GemmSpec) for g in gemms)
    # The pass class itself must be referenceable for type checks even
    # before instantiation.
    assert BlockScaleSplitKZeroInitFusionPass is not None


def test_default_pick_split_k_gate():
    """The picker acts as a yes/no gate: 0 when SplitK doesn't pay off, >0
    otherwise. The actual SplitK count is decided by AITER's runtime CSV."""
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        _default_pick_split_k,
    )

    # Tiny K -- no SplitK
    assert _default_pick_split_k(M=8, N=2048, K=1024, dtype=torch.bfloat16) == 0
    # Large M -- no SplitK (parallelism already saturated by M-tiles)
    assert _default_pick_split_k(M=1024, N=2048, K=8192, dtype=torch.bfloat16) == 0
    # K-skinny + small M -- SplitK helps
    assert _default_pick_split_k(M=8, N=2048, K=8192, dtype=torch.bfloat16) > 0


# ---------------------------------------------------------------------------
# ROCm / AITER-gated end-to-end tests
# ---------------------------------------------------------------------------


M = 8  # token count; small + K-skinny so default_pick_split_k > 0
K = 8192  # hidden dim
N = 4096  # output dim of the blockscale GEMM
GROUP_SIZE = 128
EPS = 1e-6


def _fp8_dtype() -> torch.dtype:
    """Platform-appropriate FP8 dtype.

    gfx942 (MI300X) uses ``float8_e4m3fnuz``; gfx950 (MI355X) uses
    ``float8_e4m3fn``. The test is shape-only (we never read the contents),
    so we just defer to whatever vllm's ``current_platform`` reports.
    """
    from vllm.platforms import current_platform

    return current_platform.fp8_dtype()


class _BaseP1Module(torch.nn.Module):
    """P1: per-token quant -> blockscale GEMM."""

    def __init__(self, K: int, N: int):
        super().__init__()
        self._fp8 = _fp8_dtype()
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=self._fp8), requires_grad=False
        )
        self.Bs = torch.nn.Parameter(
            torch.empty(
                (N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Producer: dynamic per-token FP8 quant -- returns (out_fp8, scale)
        out, scale = torch.ops.vllm.rocm_aiter_per_token_quant(
            x, self._fp8, None
        )
        # Consumer: CK blockscale FP8 GEMM
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _BaseP2Module(torch.nn.Module):
    """P2: Gemma RMSNorm + FP8 group quant -> blockscale GEMM."""

    def __init__(self, K: int, N: int):
        super().__init__()
        self._fp8 = _fp8_dtype()
        self.weight = torch.nn.Parameter(
            torch.ones(K, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=self._fp8), requires_grad=False
        )
        self.Bs = torch.nn.Parameter(
            torch.empty(
                (N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, scale = torch.ops.vllm.rocm_aiter_gemma_rmsnorm_fp8_group_quant(
            x, self.weight, EPS, GROUP_SIZE
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _BaseP3Module(torch.nn.Module):
    """P3: gated RMSNorm + FP8 group quant -> blockscale GEMM."""

    def __init__(self, K: int, N: int):
        super().__init__()
        self._fp8 = _fp8_dtype()
        self.weight = torch.nn.Parameter(
            torch.ones(K, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=self._fp8), requires_grad=False
        )
        self.Bs = torch.nn.Parameter(
            torch.empty(
                (N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out, scale = torch.ops.vllm.rocm_aiter_gated_rmsnorm_fp8_group_quant(
            x, z, self.weight, EPS, GROUP_SIZE
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


def _build_vllm_config(dtype: torch.dtype = torch.bfloat16) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(
                fuse_blockscale_splitk_zero_init=True,
                eliminate_noops=True,
            ),
        ),
    )


def _run_pass_on_module(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    vllm_config: VllmConfig,
):
    """Compile ``module`` with the fusion pass and return the TestBackend."""
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        BlockScaleSplitKZeroInitFusionPass,
    )

    fusion_pass = BlockScaleSplitKZeroInitFusionPass(vllm_config)
    backend = TestBackend(
        NoOpEliminationPass(vllm_config),
        fusion_pass,
        PostCleanupPass(vllm_config),
    )
    compiled = torch.compile(module, backend=backend)
    compiled(*inputs)
    return backend, fusion_pass


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required for the blockscale SplitK zero-init fusion test",
)
@pytest.mark.parametrize(
    "module_cls,extra_inputs",
    [
        (_BaseP1Module, ()),
        (_BaseP2Module, ()),
        # P3 takes an extra `z` input for the gate.
        (_BaseP3Module, ("z",)),
    ],
)
def test_pass_rewrites_producer_gemm_chain(
    module_cls: type[torch.nn.Module],
    extra_inputs: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end pattern test: each registered producer combined with
    rocm_aiter_gemm_a8w8_blockscale should be rewritten to the mutating
    ``_with_zero_init`` + ``_splitk`` pair."""
    torch._dynamo.reset()
    vllm_config = _build_vllm_config()

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        monkeypatch.context() as m,
    ):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        model = module_cls(K, N)
        x = torch.randn(M, K, dtype=torch.bfloat16)
        # Leave M static so the pattern matcher's extra_check can read a
        # concrete int and decide fuse/skip via pick_split_k. The fusion
        # itself works under dynamic shapes too, but the test exercises a
        # fixed-shape compile cycle for simplicity.
        inputs: list[torch.Tensor] = [x]
        if extra_inputs:
            # P3 expects the gate `z` with the same shape as x.
            z = torch.randn(M, K, dtype=torch.bfloat16)
            inputs.append(z)

        backend, fusion_pass = _run_pass_on_module(model, tuple(inputs), vllm_config)

        assert fusion_pass.matched_count >= 1, (
            f"Expected at least 1 fusion match for {module_cls.__name__}, "
            f"got {fusion_pass.matched_count}"
        )

        # Sanity: the rewritten graph must contain the mutating GEMM op.
        post_ops = {
            n.target for n in backend.graph_post_pass.nodes if n.op == "call_function"
        }
        assert (
            torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale_splitk.default in post_ops
            or any(
                "rocm_aiter_gemm_a8w8_blockscale_splitk" in str(t) for t in post_ops
            )
        ), "splitk GEMM op missing from rewritten graph"


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required",
)
def test_pass_no_match_when_extra_check_rejects(monkeypatch: pytest.MonkeyPatch):
    """When ``pick_split_k`` returns 0 (e.g. K is too small for SplitK to pay
    off), the fusion must NOT rewrite the chain."""
    torch._dynamo.reset()
    vllm_config = _build_vllm_config()

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        monkeypatch.context() as m,
    ):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)

        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

        small_K = 512  # below the default_pick_split_k threshold of 4096
        model = _BaseP1Module(small_K, N)
        x = torch.randn(M, small_K, dtype=torch.bfloat16)

        backend, fusion_pass = _run_pass_on_module(model, (x,), vllm_config)

        assert fusion_pass.matched_count == 0, (
            "Fusion should be skipped when SplitK doesn't pay off, but "
            f"matched_count={fusion_pass.matched_count}"
        )
