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
            x=x, weight=self.weight, variance_epsilon=EPS, group_size=GROUP_SIZE
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _BaseP3Module(torch.nn.Module):
    """P3: gated RMSNorm + FP8 group quant -> blockscale GEMM.

    The AITER ``gated_rmsnorm_fp8_group_quant`` kernel only supports
    ``head_dim == 128`` and expects 3D ``[T, H, D]`` inputs with a weight of
    shape ``[D]``. Production Qwen3-Next GDN already calls this op with the
    correct 3D layout, so the test module emits the same call shape.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        self._fp8 = _fp8_dtype()
        assert K % GROUP_SIZE == 0, "K must be a multiple of head_dim=128 for P3"
        self._num_heads = K // GROUP_SIZE
        self._head_dim = GROUP_SIZE
        self.weight = torch.nn.Parameter(
            torch.ones(self._head_dim, dtype=torch.bfloat16), requires_grad=False
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
        # Reshape (T, H*D) -> (T, H, D) for the gated rmsnorm kernel.
        x3 = x.view(-1, self._num_heads, self._head_dim)
        z3 = z.view(-1, self._num_heads, self._head_dim)
        out, scale = torch.ops.vllm.rocm_aiter_gated_rmsnorm_fp8_group_quant(
            x3, z3, self.weight, EPS, GROUP_SIZE
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


# ---------------------------------------------------------------------------
# Per-producer test modules for the *new* registered ProducerSpec entries.
# These exercise the producer plumbing (P1 HIP per-1x128, P2 / P2-add Triton
# rmsnorm-quant, P3 fused gated, P4 silu+mul + quant). Each module
# emits the *same* keyword-style call that the upstream producer fusion
# pass produces in the real Qwen3-Next FX graph, so the pattern matcher
# sees an FX node whose arg layout matches what we register.
# ---------------------------------------------------------------------------


class _NewP1GroupQuantModule(torch.nn.Module):
    """Upstream P1 producer: `rocm_aiter_group_fp8_quant(x, group_size)`.

    HIP per-1x128 group FP8 quant. Distinct from the per-token quant
    used in `_BaseP1Module`; this is the producer that actually fires in
    Qwen3-Next-class models on gfx950.
    """

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
        out, scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(x, GROUP_SIZE)
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _NewP2RMSNormGroupQuantModule(torch.nn.Module):
    """Upstream P2 producer: `rocm_aiter_rmsnorm_fp8_group_quant`.

    Calls the op with the all-kwargs convention that the upstream
    `AiterRMSFp8GroupQuantPattern` emits in the FX graph.
    """

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
        out, scale = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
            x=x,
            weight=self.weight,
            variance_epsilon=EPS,
            group_size=GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _NewP2AddRMSNormGroupQuantModule(torch.nn.Module):
    """Upstream P2-add producer: `rocm_aiter_rmsnorm_with_add_fp8_group_quant`.

    Returns three outputs (fp8, residual, scales); the fusion pass must
    preserve the residual SSA edge in the rewrite.
    """

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

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out, res, scale = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x=x,
            residual=residual,
            weight=self.weight,
            variance_epsilon=EPS,
            group_size=GROUP_SIZE,
        )
        gemm = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )
        # Keep ``res`` live so the residual SSA edge survives fusion. We
        # don't add the residual to the GEMM (shapes differ: residual is
        # (M, K), GEMM is (M, N)); use ``sum()`` to fold ``res`` into a
        # scalar and broadcast it into the GEMM output.
        return gemm + res.sum()


class _NewP3FusedRMSGatedModule(torch.nn.Module):
    """Upstream P3 producer: `rocm_aiter_fused_rms_gated_fp8_group_quant`."""

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
        out, scale = torch.ops.vllm.rocm_aiter_fused_rms_gated_fp8_group_quant(
            x=x,
            weight=self.weight,
            bias=None,
            z=z,
            eps=EPS,
            norm_before_gate=True,
            activation="silu",
            group_size=GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _NewP4ActMulModule(torch.nn.Module):
    """Upstream P4 producer: `rocm_aiter_act_mul_and_fp8_group_quant`.

    Takes x with last dim 2*K (gate_up_proj output) and applies silu*mul
    + group quant to give (M, K) FP8 feeding down_proj.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        self._fp8 = _fp8_dtype()
        # x has 2*K columns (gate_up); GEMM B is (N, K).
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
        out, scale = torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant(
            x=x,
            group_size=GROUP_SIZE,
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

        # Sanity: the rewritten graph must contain the mutating GEMM op,
        # either bare or wrapped by AOTAutograd's auto_functionalized HOP
        # (which is the default lowering for mutates_args ops).
        post_ops = {
            n.target for n in backend.graph_post_pass.nodes if n.op == "call_function"
        }
        splitk_op = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale_splitk.default
        # auto_functionalized wraps the mutating op as its first positional
        # arg; walk the FX args of every node to look for it there too.
        wrapped_targets = {
            arg
            for n in backend.graph_post_pass.nodes
            if n.op == "call_function"
            for arg in n.args
            if isinstance(arg, torch._ops.OpOverload)
        }
        assert (
            splitk_op in post_ops
            or splitk_op in wrapped_targets
            or any(
                "rocm_aiter_gemm_a8w8_blockscale_splitk" in str(t) for t in post_ops
            )
        ), "splitk GEMM op missing from rewritten graph"


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required for the blockscale SplitK zero-init fusion test",
)
@pytest.mark.parametrize(
    "module_cls,extra_inputs,expected_replaced_op",
    [
        (
            _NewP1GroupQuantModule,
            (),
            "rocm_aiter_group_fp8_quant_with_zero_init",
        ),
        (
            _NewP2RMSNormGroupQuantModule,
            (),
            "rocm_aiter_rmsnorm_fp8_group_quant_with_zero_init",
        ),
        (
            _NewP2AddRMSNormGroupQuantModule,
            ("residual",),
            "rocm_aiter_rmsnorm_with_add_fp8_group_quant_with_zero_init",
        ),
        (
            _NewP3FusedRMSGatedModule,
            ("z",),
            "rocm_aiter_fused_rms_gated_fp8_group_quant_with_zero_init",
        ),
        (
            _NewP4ActMulModule,
            ("gate_up_2k",),  # special-cased input: x has 2*K cols
            "rocm_aiter_act_mul_and_fp8_group_quant_with_zero_init",
        ),
    ],
)
def test_pass_rewrites_new_producer_specs(
    module_cls: type[torch.nn.Module],
    extra_inputs: tuple[str, ...],
    expected_replaced_op: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each of the new ProducerSpec entries must rewrite its producer
    -> blockscale-GEMM chain into the ``_with_zero_init`` + ``_splitk``
    pair. The test asserts ``matched_count >= 1`` (i.e. the FX rewriter
    actually fired) and verifies the post-pass graph contains the
    expected mutating producer op.
    """
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
        inputs: list[torch.Tensor] = [x]
        if "residual" in extra_inputs:
            inputs.append(torch.randn(M, K, dtype=torch.bfloat16))
        if "z" in extra_inputs:
            inputs.append(torch.randn(M, K, dtype=torch.bfloat16))
        if "gate_up_2k" in extra_inputs:
            # P4: replace x with (M, 2*K) input (silu+mul halves it back to K).
            inputs = [torch.randn(M, 2 * K, dtype=torch.bfloat16)]

        backend, fusion_pass = _run_pass_on_module(model, tuple(inputs), vllm_config)

        assert fusion_pass.matched_count >= 1, (
            f"Expected at least 1 fusion match for {module_cls.__name__}, "
            f"got {fusion_pass.matched_count}"
        )

        # The mutating producer / GEMM ops live inside ``auto_functionalized``
        # nodes (PyTorch's HOP that functionalizes ops with mutable args).
        # We have to inspect the first positional arg of each
        # ``auto_functionalized`` node, not the node target itself, to verify
        # the rewrite landed.
        from torch._higher_order_ops.auto_functionalize import auto_functionalized

        functionalized_targets: set[str] = set()
        for n in backend.graph_post_pass.nodes:
            if n.op != "call_function":
                continue
            if n.target is auto_functionalized and n.args:
                functionalized_targets.add(str(n.args[0]))
            else:
                functionalized_targets.add(str(n.target))

        assert any(expected_replaced_op in s for s in functionalized_targets), (
            f"Expected post-pass graph to contain {expected_replaced_op!r}; "
            f"got targets: {sorted(functionalized_targets)}"
        )
        assert any(
            "rocm_aiter_gemm_a8w8_blockscale_splitk" in s
            for s in functionalized_targets
        ), "splitk GEMM op missing from rewritten graph"


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required for the per-pair attribution test",
)
def test_per_pair_attribution_counts(monkeypatch: pytest.MonkeyPatch):
    """The fusion pass exposes per-(producer, gemm) attribution via
    ``_count_per_pair(graph)``. The counter walks the post-apply graph and
    pairs each ``auto_functionalized(producer.with_zero_init_op)`` HOP with
    its downstream ``auto_functionalized(gemm.splitk_op)`` consumer.

    This verifies the attribution keys on the (producer, gemm) pair rather
    than on ``producer.with_zero_init_op`` alone. Keying on the producer op
    alone would attribute every (producer, *) match to the last-registered
    ``gemm`` overload in the registry (the triton variant would absorb every
    CK match), producing misleading per-pair logs even though it leaves
    ``matched_count`` unchanged.
    """
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

        model = _NewP1GroupQuantModule(K, N)
        x = torch.randn(M, K, dtype=torch.bfloat16)
        backend, fusion_pass = _run_pass_on_module(model, (x,), vllm_config)

        counts = fusion_pass._count_per_pair(backend.graph_post_pass)
        ck_key = "aiter_group_fp8_quant__x__aiter_gemm_a8w8_blockscale"
        triton_key = "aiter_group_fp8_quant__x__aiter_triton_gemm_a8w8_blockscale"
        # Production graph uses the legacy CK GEMM (the only one the
        # test module instantiates), so the CK-side pair gets the credit.
        assert counts.get(ck_key, 0) == 1, (
            f"Expected exactly 1 match attributed to {ck_key}, got {counts}"
        )
        # The triton variant shares the producer.with_zero_init_op overload
        # with the CK variant; mis-keying the attribution would show up as
        # triton_key=1 / ck_key=0.
        assert counts.get(triton_key, 0) == 0, (
            f"Triton-gemm key {triton_key} should be 0 for a CK-only graph "
            f"(would indicate mis-keyed per-pair attribution), got {counts}"
        )
        # Sanity: the total per-pair count must equal the global match
        # count tracked by the pattern matcher pass.
        total_per_pair = sum(counts.values())
        assert total_per_pair == fusion_pass.matched_count, (
            f"per-pair total {total_per_pair} != matched_count "
            f"{fusion_pass.matched_count}; counts={counts}"
        )


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
