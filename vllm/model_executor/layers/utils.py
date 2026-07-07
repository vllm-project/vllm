# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

import functools
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.flashinfer import (
    flashinfer_bf16_mm_impl,
    is_flashinfer_bf16_gemm_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties

    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)


# Plan/run dispatch: resolve the tuned (runner, tactic) per weight once,
# then make the per-call path a bucket lookup plus a direct kernel launch --
# no mm_bf16 wrapper, no backend heuristic, no autotuner query per call.
_MM_BF16_PLAN_TABLES: dict = {}  # (n,k,dtype,bias_is_none,pdl,dev) -> [(runner,tactic)]
_MM_BF16_PLANS: dict = {}  # (weight_ptr, bias_is_none, pdl) -> _MmBf16GemmPlan


def _closure_var(bound_method, name: str):
    fn = bound_method.__func__
    for var, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if var == name:
            return cell.cell_contents
    raise KeyError(name)


class _MmBf16GemmPlan:
    __slots__ = ("weight_t", "workspace", "num_buckets", "lanes", "pdl")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, pdl: bool):
        import flashinfer.gemm.gemm_base as gb

        n, k = weight.shape
        self.weight_t = weight.t()
        self.pdl = pdl
        self.workspace = gb._get_cache_buf(
            "mm_bf16_workspace", gb.DEFAULT_WORKSPACE_SIZE, weight.device
        )
        key = (n, k, weight.dtype, bias is None, pdl, weight.device.index)
        table = _MM_BF16_PLAN_TABLES.get(key)
        if table is None:
            table = _MM_BF16_PLAN_TABLES[key] = self._tune(weight, bias, pdl)
        self.num_buckets = len(table)
        # A2-full: pre-bind one launch lane per bucket, resolving everything
        # runner.forward would redo per call (handle/algo/graph lookups).
        self.lanes = [self._make_lane(r, t, gb) for r, t in table]

    def _make_lane(self, runner, tactic: int, gb):
        name = type(runner).__name__
        wt = self.weight_t
        w_nk = wt.transpose(-2, -1)
        ws = self.workspace
        if name == "CutlassBf16GemmRunner":
            module = _closure_var(runner.forward, "module")
            t = tactic

            def lane(x, bias, out):
                module.bf16_gemm(x, w_nk, out, ws, t)

            return lane
        if name == "CublasltBf16GemmRunner":
            module = _closure_var(runner.forward, "module")
            handle = torch.cuda.current_blas_handle()
            t = max(tactic, 0)
            get_algos = runner._get_algos
            algo_cache: dict = {}

            def lane(x, bias, out):
                m = x.shape[0]
                entry = algo_cache.get(m)
                if entry is None:
                    algo_buf, count = get_algos([x, wt, bias, False, out, ws])
                    entry = algo_cache[m] = (algo_buf, t if t < count else 0)
                module.mm_bf16_cublaslt_run_with_algo(
                    x, w_nk, out, ws, handle, entry[0], entry[1]
                )

            return lane
        if name == "CudnnBf16GemmRunner" and runner._use_override_shape:
            t = max(tactic, 0)
            get_graph = runner._get_override_graph
            exec_fn = gb.execute_cudnn_gemm_bf16_graph_override_shape
            graph_cache: dict = {}

            def lane(x, bias, out):
                m = x.shape[0]
                g = graph_cache.get(m)
                if g is None:
                    g = graph_cache[m] = get_graph(x, wt, bias, out)
                exec_fn(g, x, wt, bias, out, ws, tactic=t)

            return lane

        pdl = self.pdl

        def lane(x, bias, out):
            runner(inputs=[x, wt, bias, pdl, out, ws], tactic=tactic)

        return lane

    @staticmethod
    def _tune(weight: torch.Tensor, bias: torch.Tensor | None, pdl: bool):
        import os

        import flashinfer.gemm.gemm_base as gb
        from flashinfer.autotuner import AutoTuner, autotune
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

        n, k = weight.shape
        dev = weight.device
        max_m = int(os.environ.get("VLLM_BF16_PLAN_MAX_M", "256"))
        buckets = get_hybrid_num_tokens_buckets(max_m)
        weight_t = weight.t()
        ws = gb._get_cache_buf("mm_bf16_workspace", gb.DEFAULT_WORKSPACE_SIZE, dev)
        runners = []
        for build in (
            lambda: gb._cudnn_gemm_bf16_runner(is_a_k_major=True, is_b_k_major=True),
            lambda: gb.get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner(),
            lambda: gb.get_gemm_sm100_module_cutlass_bf16().cutlass_bf16_gemm_runner(),
            lambda: gb._tgv_gemm_runner(weight.dtype, gb.is_sm100f_supported(dev)),
        ):
            try:
                runners.append(build())
            except Exception:
                continue
        tuner = AutoTuner.get()
        cfg = gb._BF16_GEMM_SM100_TUNING_CONFIG
        # Tuning the largest bucket profiles every smaller bucket in the same
        # pass; the per-bucket queries below read the winners back.
        with autotune(True):
            x = torch.empty(buckets[-1], k, dtype=weight.dtype, device=dev)
            out = torch.empty(buckets[-1], n, dtype=weight.dtype, device=dev)
            tuner.choose_one("bf16_gemm", runners, cfg, [x, weight_t, bias, pdl, out, ws])
        table = []
        for b in buckets:
            xb = torch.empty(b, k, dtype=weight.dtype, device=dev)
            ob = torch.empty(b, n, dtype=weight.dtype, device=dev)
            table.append(
                tuner.choose_one("bf16_gemm", runners, cfg, [xb, weight_t, bias, pdl, ob, ws])
            )
        logger.info_once("Built mm_bf16 plan tables (A2 prototype).")
        return table

    def run(
        self,
        x_2d: torch.Tensor,
        bias: torch.Tensor | None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        idx = (x_2d.shape[0] - 1).bit_length()
        if idx >= self.num_buckets:
            idx = self.num_buckets - 1
        self.lanes[idx](x_2d, bias, out)
        return out


def cuda_flashinfer_bf16_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool = False,
) -> torch.Tensor:
    """Route an unquantized BF16 matmul through FlashInfer's ``mm_bf16``.

    Unsupported dtype/device combinations fail explicitly instead of silently
    mixing model backends.
    """
    if x.dim() == 0 or weight.dim() != 2:
        return torch.nn.functional.linear(x, weight, bias)

    # Fast path: a plan keyed on this exact weight/bias/pdl already validated
    # dtypes and devices when it was built.
    plan = _MM_BF16_PLANS.get((weight.data_ptr(), bias is None, pdl))
    if plan is not None:
        K = x.shape[-1]
        M = x.numel() // K
        N = weight.shape[0]
        x_2d = x.reshape(M, K)
        out_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)
        plan.run(x_2d, bias, out_2d)
        return out_2d.reshape(*x.shape[:-1], N)

    K = x.shape[-1]
    M = x.numel() // K if K > 0 else 0
    N = weight.shape[0]
    bias_ok = bias is None or (
        bias.dtype == torch.bfloat16
        and bias.is_cuda
        and bias.device == x.device
        and bias.dim() == 1
        and bias.shape[0] == N
    )
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not weight.is_cuda
        or weight.device != x.device
        or M == 0
        or N == 0
        or K == 0
        or not bias_ok
    ):
        logger.warning_once("Using default unquantized gemm (torch).")
        return torch.nn.functional.linear(x, weight, bias)

    if torch.cuda.is_current_stream_capturing():
        # Plans are built during the warmup dummy run; never tune inside
        # graph capture.
        return torch.nn.functional.linear(x, weight, bias)
    plan = _MM_BF16_PLANS[(weight.data_ptr(), bias is None, pdl)] = _MmBf16GemmPlan(
        weight, bias, pdl
    )
    x_2d = x.reshape(M, K)
    out_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)
    plan.run(x_2d, bias, out_2d)
    return out_2d.reshape(*x.shape[:-1], N)


def cuda_flashinfer_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pdl: bool = False,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16)


def cuda_flashinfer_bf16_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    pdl: bool = False,
) -> torch.Tensor:
    return torch.ops.vllm.cuda_flashinfer_bf16_gemm(x, weight, bias, pdl)


direct_register_custom_op(
    op_name="cuda_flashinfer_bf16_gemm",
    op_func=cuda_flashinfer_bf16_gemm_impl,
    fake_impl=cuda_flashinfer_bf16_gemm_fake,
)


def use_aiter_triton_gemm(n, m, k, dtype):
    if (
        not rocm_aiter_ops.is_triton_gemm_enabled()
        # MI300's - fp8nuz=True
        or current_platform.is_fp8_fnuz()
        or dtype not in [torch.float16, torch.bfloat16]
    ):
        return False

    # use hipblaslt for the larger GEMMs
    if n > 2048 and m > 512:
        return False
    return (
        (m == 5120 and k == 2880)
        or (m == 2880 and k == 4096)
        or (m == 128 and k == 2880)
        or (m == 640 and k == 2880)
        or (m == 2880 and k == 512)
    )


def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950

    n = x.numel() // x.size(-1)
    m = weight.shape[0]
    k = weight.shape[1]

    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # candidate for atomic reduce count splitk?
    fits_wvsplitkrc = (
        N_p2 * m * ((k + 512 - 1) // 512)
    ) <= 128 * 1024 * 12  # deterministic
    fits_wvsplitkrc &= CuNeeded <= cu_count

    use_skinny_reduce_counting = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and on_gfx950()
        and x.dtype in [torch.float16, torch.bfloat16]
        and x.dim() == 2
        and (
            10 <= n <= 128
            and k % 8 == 0
            and k > 512
            and m % 16 == 0
            and fits_wvsplitkrc
            and weight.is_contiguous()
        )
    )

    if use_skinny_reduce_counting:
        return ops.wvSplitKrc(x, weight, cu_count, bias)

    if use_aiter_triton_gemm(n, m, k, x.dtype):
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return gemm_a16w16(x, weight, bias)

    use_skinny = (
        envs.VLLM_ROCM_USE_SKINNY_GEMM
        and (on_gfx9() or on_gfx1x())
        and x.dtype in [torch.float16, torch.bfloat16]
        and k % 8 == 0
    )

    if use_skinny:
        x_view = x.reshape(-1, x.size(-1))
        if m > 8 and 0 < n <= 5:
            cu_count = num_compute_units()
            out = ops.wvSplitK(weight, x_view, cu_count, bias)
            return out.reshape(*x.shape[:-1], weight.shape[0])
        elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
            out = ops.LLMM1(weight, x_view, 4)
            return out.reshape(*x.shape[:-1], weight.shape[0])

    if rocm_aiter_ops.is_tgemm_enabled():
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, weight, bias)

    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm",
    op_func=rocm_unquantized_gemm_impl,
    fake_impl=rocm_unquantized_gemm_fake,
)


def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool:
    return (
        torch.cpu._is_amx_tile_supported()
        and (dtype in (torch.bfloat16, torch.int8))
        and k % 32 == 0
        and n % 16 == 0
    )


def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module,
    remove_weight: bool,
) -> None:
    # skip for missing layers
    if layer.weight.is_meta:
        layer.cpu_linear = torch.nn.functional.linear
        return

    if layer.weight.ndim != 2:
        # this is not a linear layer
        # For now it should be a causal_conv1d op
        if torch.cpu._is_amx_tile_supported():
            # prepack conv weight
            layer.weight.data = ops.causal_conv1d_weight_pack(
                layer.weight.view(
                    layer.weight.size(0),
                    layer.weight.size(2),
                )
            )
        return

    N, K = layer.weight.size()
    dtype = layer.weight.dtype

    # Zen CPU path: zentorch_linear_unary with optional eager weight prepacking.
    if current_platform.is_zen_cpu() and hasattr(
        torch.ops.zentorch, "zentorch_linear_unary"
    ):
        zen_weight = layer.weight.detach()
        is_prepacked = False

        if envs.VLLM_ZENTORCH_WEIGHT_PREPACK and hasattr(
            torch.ops.zentorch, "zentorch_weight_prepack_for_linear"
        ):
            zen_weight = torch.ops.zentorch.zentorch_weight_prepack_for_linear(
                zen_weight
            )
            is_prepacked = True

        layer.cpu_linear = lambda x, weight, bias, _p=is_prepacked: (
            torch.ops.zentorch.zentorch_linear_unary(
                x, zen_weight, bias, is_weight_prepacked=_p
            )
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using zentorch_linear_unary (prepacked=%s)",
            is_prepacked,
        )
        return

    if envs.VLLM_CPU_SGL_KERNEL and check_cpu_sgl_kernel(N, K, dtype):
        packed_weight = torch.ops._C.convert_weight_packed(layer.weight)
        if getattr(layer, "bias", None) is not None:
            bias_f32 = layer.bias.to(torch.float32)
        else:
            bias_f32 = None
        layer.cpu_linear = lambda x, weight, bias: torch.ops._C.weight_packed_linear(
            x, packed_weight, bias_f32 if bias is not None else None, True
        )
        if remove_weight:
            layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        logger.debug_once(
            "CPU unquantized GEMM dispatch: using sgl-kernel weight_packed_linear"
        )
        return
    elif (
        ops._supports_onednn
        and current_platform.get_cpu_architecture() != CpuArchEnum.POWERPC
    ):
        try:
            origin_weight = layer.weight
            handler = ops.create_onednn_mm(origin_weight.t(), 32)
            layer.cpu_linear = lambda x, weight, bias: ops.onednn_mm(handler, x, bias)
            if remove_weight:
                layer.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            logger.debug_once("CPU unquantized GEMM dispatch: using oneDNN onednn_mm")
            return
        except RuntimeError as e:
            logger.warning_once(
                "Failed to create oneDNN linear, fallback to torch linear."
                f" Exception: {e}"
            )

    # fallback case
    layer.cpu_linear = lambda x, weight, bias: torch.nn.functional.linear(
        x, weight, bias
    )
    logger.debug_once(
        "CPU unquantized GEMM dispatch: using torch.nn.functional.linear (fallback)"
    )


def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return layer.cpu_linear(x, weight, bias)


def _get_bf16_linear_config() -> tuple[str, bool]:
    """Read the BF16 linear settings from the active VllmConfig.

    Returns the Torch backend with PDL disabled when there is no active
    VllmConfig, e.g. during unit tests that don't set up an engine.
    """
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()

    except (AssertionError, AttributeError, ImportError):
        return "torch", False

    if vllm_config is None or vllm_config.kernel_config is None:
        return "torch", False

    kernel_config = vllm_config.kernel_config
    backend = kernel_config.bf16_linear_backend
    if backend == "torch":
        return "torch", False

    if not is_flashinfer_bf16_gemm_supported():
        logger.warning_once(
            "FlashInfer mm_bf16 is unavailable or unsupported; falling back to torch."
        )
        return "torch", False

    return backend, kernel_config.enable_bf16_pdl


def select_unquantized_gemm_impl() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        gemm_impl = rocm_unquantized_gemm
        gemm_name = "rocm_unquantized_gemm"
    elif current_platform.is_cpu():
        gemm_impl = cpu_unquantized_gemm
        gemm_name = "cpu_unquantized_gemm"
    else:
        bf16_linear_backend, enable_bf16_pdl = _get_bf16_linear_config()
        if bf16_linear_backend == "torch":
            gemm_impl = default_unquantized_gemm
            gemm_name = "torch_linear"
        else:
            gemm_impl = functools.partial(
                cuda_flashinfer_bf16_gemm,
                pdl=enable_bf16_pdl,
            )
            gemm_name = f"flashinfer_bf16_gemm(pdl={enable_bf16_pdl})"

    logger.info_once(
        "Bound %s for unquantized GEMMs.",
        gemm_name,
    )
    return gemm_impl
