# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockScaleSplitKZeroInitFusionPass``.

The pass rewrites a chain of
    (functional producer) -> (functional blockscale GEMM)
into
    torch.empty -> auto_functionalized(producer_with_zero_init, gemm_out_zero_init=Y)
                -> auto_functionalized(gemm_out, output=Y, y_is_zeroed=True)

The tests below exercise this rewrite on toy modules covering producers in
the default registry (Gemma RMSNorm, gated RMSNorm, fused GDN gated RMSNorm,
and the HIP/Triton group-quant / rmsnorm / silu-mul producers). All tests are
gated behind ``is_aiter_found_and_supported`` because they call ROCm AITER ops
at trace-time.
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
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# Always-on smoke tests of the registry module (no ROCm required)
# ---------------------------------------------------------------------------


def test_registries_module_importable():
    """The fusion pass module must be importable on all platforms."""
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        BlockScaleSplitKZeroInitFusionPass,
        GemmSpec,
        ProducerSpec,
        build_default_registries,
    )

    # The pass class itself must be referenceable for type checks even
    # before instantiation.
    assert BlockScaleSplitKZeroInitFusionPass is not None
    if not is_aiter_found_and_supported():
        return
    rocm_aiter_ops.register_ops_once()
    producers, gemms = build_default_registries()
    assert isinstance(producers, list)
    assert isinstance(gemms, list)
    assert all(isinstance(p, ProducerSpec) for p in producers)
    assert all(isinstance(g, GemmSpec) for g in gemms)


# ---------------------------------------------------------------------------
# ROCm / AITER-gated end-to-end tests
# ---------------------------------------------------------------------------


M = 8  # token count; small + K-skinny (default K-gate accepts K >= 2048)
K = 8192  # hidden dim
N = 4096  # output dim of the blockscale GEMM
GROUP_SIZE = 128
EPS = 1e-6


class _GemmaRMSNormGroupQuantModule(torch.nn.Module):
    """Gemma RMSNorm + FP8 group quant -> blockscale GEMM."""

    def __init__(self, K: int, N: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(K, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, scale = torch.ops.vllm.rocm_aiter_gemma_rmsnorm_fp8_group_quant(
            x=x,
            weight=self.weight,
            variance_epsilon=self.eps,
            group_size=GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _GatedRMSNormGroupQuantModule(torch.nn.Module):
    """Gated RMSNorm + FP8 group quant -> blockscale GEMM.

    The AITER ``gated_rmsnorm_fp8_group_quant`` kernel only supports
    ``head_dim == 128`` and expects 3D ``[T, H, D]`` inputs with a weight of
    shape ``[D]``. Production Qwen3-Next GDN already calls this op with the
    correct 3D layout, so the test module emits the same call shape.
    """

    def __init__(self, K: int, N: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        assert K % GROUP_SIZE == 0, (
            "K must be a multiple of head_dim=128 for gated RMSNorm group quant"
        )
        self._num_heads = K // GROUP_SIZE
        self._head_dim = GROUP_SIZE
        self.weight = torch.nn.Parameter(
            torch.ones(self._head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Reshape (T, H*D) -> (T, H, D) for the gated rmsnorm kernel.
        x3 = x.view(-1, self._num_heads, self._head_dim)
        z3 = z.view(-1, self._num_heads, self._head_dim)
        out, scale = torch.ops.vllm.rocm_aiter_gated_rmsnorm_fp8_group_quant(
            x3,
            z3,
            self.weight,
            self.eps,
            GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _FusedRMSGatedFP8GroupQuantModule(torch.nn.Module):
    """Fused gated RMSNorm + FP8 group quant -> blockscale GEMM.

    Mirrors the upstream ``AiterRMSNormGatedFp8GroupQuantPattern``
    replacement: call the fused op on per-head ``[T * H, D]`` tensors, then
    reshape FP8/scales back to the blockscale GEMM's ``[T, H * D]`` layout.
    """

    def __init__(self, K: int, N: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        assert K % GROUP_SIZE == 0, (
            "K must be a multiple of head_dim=128 for fused gated RMS group quant"
        )
        self._hidden_dim = K
        self._num_heads = K // GROUP_SIZE
        self._head_dim = GROUP_SIZE
        self.weight = torch.nn.Parameter(
            torch.ones(self._head_dim, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self._head_dim)
        z_flat = z.reshape(-1, self._head_dim)
        out, scale = torch.ops.vllm.rocm_aiter_fused_rms_gated_fp8_group_quant(
            x=x_flat,
            weight=self.weight,
            bias=None,
            z=z_flat,
            eps=self.eps,
            norm_before_gate=True,
            activation="silu",
            group_size=GROUP_SIZE,
        )
        out = out.reshape(-1, self._hidden_dim)
        scale = scale.reshape(-1, self._num_heads)
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


# ---------------------------------------------------------------------------
# Per-producer test modules for the *new* registered ProducerSpec entries.
# These exercise the producer-to-GEMM rewrite coverage for per-group quant,
# RMSNorm group quant, residual-add RMSNorm group quant, and silu+mul group
# quant. Each module
# emits the *same* keyword-style call that the upstream producer fusion
# pass produces in the real Qwen3-Next FX graph, so the pattern matcher
# sees an FX node whose arg layout matches what we register.
# ---------------------------------------------------------------------------


class _GroupQuantModule(torch.nn.Module):
    """Upstream producer: `rocm_aiter_group_fp8_quant(x, group_size)`.

    HIP per-1x128 group FP8 quant -- the producer that actually fires in
    Qwen3-Next-class models on gfx950.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(
            x,
            GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _RMSNormGroupQuantModule(torch.nn.Module):
    """Upstream producer: `rocm_aiter_rmsnorm_fp8_group_quant`.

    Calls the op with the all-kwargs convention that the upstream
    `AiterRMSFp8GroupQuantPattern` emits in the FX graph.
    """

    def __init__(self, K: int, N: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(K, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, scale = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant(
            x=x,
            weight=self.weight,
            variance_epsilon=self.eps,
            group_size=GROUP_SIZE,
        )
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
            out, self.B, scale, self.Bs, torch.bfloat16
        )


class _RMSNormWithAddGroupQuantModule(torch.nn.Module):
    """Upstream producer: `rocm_aiter_rmsnorm_with_add_fp8_group_quant`.

    Returns three outputs (fp8, residual, scales); the fusion pass must
    preserve the residual SSA edge in the rewrite.
    """

    def __init__(self, K: int, N: int, eps: float = EPS):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(K, dtype=torch.bfloat16), requires_grad=False
        )
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out, res, scale = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant(
            x=x,
            residual=residual,
            weight=self.weight,
            variance_epsilon=self.eps,
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


class _ActMulGroupQuantModule(torch.nn.Module):
    """Upstream producer: `rocm_aiter_act_mul_and_fp8_group_quant`.

    Takes x with last dim 2*K (gate_up_proj output) and applies silu*mul
    + group quant to give (M, K) FP8 feeding down_proj.
    """

    def __init__(self, K: int, N: int):
        super().__init__()
        # x has 2*K columns (gate_up); GEMM B is (N, K).
        self.B = torch.nn.Parameter(
            torch.empty((N, K), dtype=current_platform.fp8_dtype()),
            requires_grad=False,
        )
        self.Bs = torch.nn.Parameter(
            torch.empty((N, (K + GROUP_SIZE - 1) // GROUP_SIZE), dtype=torch.float32),
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
    execute_compiled: bool = True,
):
    """Compile ``module`` with the fusion pass and return the TestBackend."""
    from vllm.compilation.passes.fusion.blockscale_splitk_zero_init import (
        BlockScaleSplitKZeroInitFusionPass,
    )

    class _GraphOnlyBackend(TestBackend):

        def __call__(self, graph, example_inputs):
            super().__call__(graph, example_inputs)
            return lambda *args: (
                torch.empty((M, N), device=args[0].device, dtype=torch.bfloat16),
            )

    fusion_pass = BlockScaleSplitKZeroInitFusionPass(vllm_config)
    backend_cls = TestBackend if execute_compiled else _GraphOnlyBackend
    backend = backend_cls(
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
        (_GemmaRMSNormGroupQuantModule, ()),
        # The gated RMSNorm producer takes an extra `z` input for the gate.
        (_GatedRMSNormGroupQuantModule, ("z",)),
    ],
)
def test_pass_rewrites_producer_gemm_chain(
    module_cls: type[torch.nn.Module],
    extra_inputs: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end pattern test: each registered producer combined with
    rocm_aiter_gemm_a8w8_blockscale should be rewritten to the mutating
    producer + GEMM-out pair."""
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
        # M is static here just for a simple fixed-shape compile cycle; the
        # extra_check only gates on the statically known K, so the fusion
        # works under dynamic shapes too.
        inputs: list[torch.Tensor] = [x]
        if extra_inputs:
            # The gated RMSNorm producer expects `z` with the same shape as x.
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
        out_op = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale_out.default
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
            out_op in post_ops
            or out_op in wrapped_targets
            or any("rocm_aiter_gemm_a8w8_blockscale_out" in str(t) for t in post_ops)
        ), "GEMM out op missing from rewritten graph"


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required for the blockscale SplitK zero-init fusion test",
)
@pytest.mark.parametrize(
    "module_cls,extra_inputs,expected_replaced_op,execute_compiled",
    [
        (
            _GroupQuantModule,
            (),
            "rocm_aiter_group_fp8_quant_with_zero_init",
            True,
        ),
        (
            _RMSNormGroupQuantModule,
            (),
            "rocm_aiter_rmsnorm_fp8_group_quant_with_zero_init",
            True,
        ),
        (
            _RMSNormWithAddGroupQuantModule,
            ("residual",),
            "rocm_aiter_rmsnorm_with_add_fp8_group_quant_with_zero_init",
            True,
        ),
        (
            _ActMulGroupQuantModule,
            ("gate_up_2k",),  # special-cased input: x has 2*K cols
            "rocm_aiter_act_mul_and_fp8_group_quant_with_zero_init",
            True,
        ),
        (
            _FusedRMSGatedFP8GroupQuantModule,
            ("z",),
            "rocm_aiter_fused_rms_gated_fp8_group_quant_with_zero_init",
            # This case validates the reshape-aware graph rewrite; launching
            # the tiny synthetic GEMM shape can hit AITER runtime/JIT instability.
            False,
        ),
    ],
)
def test_pass_rewrites_new_producer_specs(
    module_cls: type[torch.nn.Module],
    extra_inputs: tuple[str, ...],
    expected_replaced_op: str,
    execute_compiled: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each of the new ProducerSpec entries must rewrite its producer
    -> blockscale-GEMM chain. The test asserts ``matched_count >= 1`` and
    verifies the post-pass graph contains the expected producer op.
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
            # silu+mul consumes (M, 2*K) and halves the last dim back to K.
            inputs = [torch.randn(M, 2 * K, dtype=torch.bfloat16)]

        backend, fusion_pass = _run_pass_on_module(
            model,
            tuple(inputs),
            vllm_config,
            execute_compiled=execute_compiled,
        )

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
            "rocm_aiter_gemm_a8w8_blockscale_out" in s
            for s in functionalized_targets
        ), "GEMM out op missing from rewritten graph"


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required",
)
def test_pass_rewrites_rmsnorm_group_quant_with_1e_5_eps(
    monkeypatch: pytest.MonkeyPatch,
):
    """The RMSNorm producer patterns cover common vLLM eps values."""
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

        model = _RMSNormGroupQuantModule(K, N, eps=1e-5)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        _, fusion_pass = _run_pass_on_module(model, (x,), vllm_config)

        assert fusion_pass.matched_count >= 1, (
            "Fusion should match RMSNorm group quant with eps=1e-5, but "
            f"matched_count={fusion_pass.matched_count}"
        )


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required",
)
def test_pass_no_match_when_extra_check_rejects(monkeypatch: pytest.MonkeyPatch):
    """When K is below the configured K-gate (too small for SplitK to pay off),
    the fusion's extra_check must reject the match and NOT rewrite the chain."""
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

        small_K = 512  # below the default configured K-gate (K < 2048)
        model = _GroupQuantModule(small_K, N)
        x = torch.randn(M, small_K, dtype=torch.bfloat16)

        _, fusion_pass = _run_pass_on_module(model, (x,), vllm_config)

        assert fusion_pass.matched_count == 0, (
            "Fusion should be skipped when SplitK doesn't pay off, but "
            f"matched_count={fusion_pass.matched_count}"
        )


@pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCm + AITER required",
)
def test_pass_respects_min_k_env_override(monkeypatch: pytest.MonkeyPatch):
    """Lowering the configured K-gate allows otherwise-rejected small-K GEMMs."""
    torch._dynamo.reset()

    with monkeypatch.context() as m:
        small_K = 512
        m.setenv(
            "VLLM_ROCM_AITER_BLOCKSCALE_SPLITK_ZERO_INIT_MIN_K",
            str(small_K),
        )
        vllm_config = _build_vllm_config()

        pass_config = vllm_config.compilation_config.pass_config
        assert pass_config.blockscale_splitk_zero_init_min_k == small_K

        with vllm.config.set_current_vllm_config(vllm_config):
            torch.set_default_device("cuda")
            torch.set_default_dtype(torch.bfloat16)
            torch.manual_seed(0)

            m.setenv("VLLM_ROCM_USE_AITER", "1")
            rocm_aiter_ops.refresh_env_variables()

            model = _GroupQuantModule(small_K, N)
            x = torch.randn(M, small_K, dtype=torch.bfloat16)

            _, fusion_pass = _run_pass_on_module(model, (x,), vllm_config)

            assert fusion_pass.matched_count == 1, (
                "Fusion should honor the configured K threshold, but "
                f"matched_count={fusion_pass.matched_count}"
            )
