# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's BF16 linears on the decode GEMM engines.

The draft step multiplies `R * k` rows (`R * (k + 1)` for the context
projections) by 0.7 GB (TP8 rank) to 7 GB (DEP rank) of BF16 weights: a decode
GEMM whose time is the weight stream. cuBLAS runs these skinny shapes through
split-K kernels at 2.4-3.6 TB/s (Phase-1 drafter breakdown); the warp split-K of
`cutedsl/gemm/warp_splitk.py` (many small co-resident CTAs, one cp.async weight
ring each, mma.sync, FP32 split-K partials reduced in shared memory, one BF16
rounding) keeps the stream near HBM speed up to 32 rows.

`DraftGemm` picks the engine per row count (`WARP_ROWS`), compiles it the
first time an eager call sees that count (the CUDA-graph warmup runs every
captured size eagerly before its capture) and runs cuBLAS for anything else.
`install` swaps the drafter linears' unquantized `apply` for it. Precision:
BF16 operands, FP32 accumulation, one BF16 rounding, as cuBLAS; only the
summation order differs.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Row counts up to which the warp split-K runs (eligible shapes: N % 16 == 0,
# K <= 8192 in 128 tiles or K <= 16384 in 256 tiles); cuBLAS past it.
WARP_ROWS = 32


def warp_eligible(n, k):
    """Shapes the warp split-K takes: 16-row output tiles and at most 64 K tiles
    (128 wide, or 256 wide when K is a multiple of 256: the TP1 down projection's
    K = 14336 is 56 tiles of 256)."""
    if n % 16 or k % 128:
        return False
    return k // 128 <= 64 or (k % 256 == 0 and k // 256 <= 64)


class DraftGemm:
    """`out[M, N] = x[M, K] @ weight[N, K]^T` with an engine per prepared M."""

    def __init__(self, weight):
        self.weight = weight
        n, k = weight.shape
        self.n, self.k = n, k
        self.sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.eligible = (
            weight.dtype == torch.bfloat16
            and weight.stride() == (k, 1)
            and weight.data_ptr() % 32 == 0
            and warp_eligible(n, k)
        )
        self.engines = {}

    def kind(self, m):
        return "warp" if m in self.engines else "cublas"

    def prepare(self, m):
        """Compile the engine of `m` rows (host only; never under capture)."""
        if m in self.engines or not self.eligible or not 1 <= m <= WARP_ROWS:
            return
        from ..cutedsl.gemm.decode import warp_tactic
        from ..cutedsl.gemm.warp_splitk import PreparedWarpSplitK

        try:
            tactic = warp_tactic(m, self.n, self.k, self.sms)
            self.engines[m] = PreparedWarpSplitK(self.weight, {m: tactic})
        except Exception as exc:  # capability fallback: cuBLAS
            logger.warning_once(
                "DSpark drafter GEMM %dx%d (M %d) on cuBLAS: %s", self.n, self.k, m, exc
            )
            self.eligible = False

    def run(self, x, out):
        m = x.shape[0]
        engine = self.engines.get(m)
        if engine is None and not torch.cuda.is_current_stream_capturing():
            self.prepare(m)
            engine = self.engines.get(m)
        if (
            engine is None
            or x.stride() != (self.k, 1)
            or x.data_ptr() % 32
            or out.stride() != (self.n, 1)
            or out.data_ptr() % 32
        ):
            return torch.mm(x, self.weight.t(), out=out)
        return engine.run(x, out)


def _linear_method_base():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    return UnquantizedLinearMethod


class DraftLinearMethod(_linear_method_base()):  # type: ignore[misc]
    """An unquantized linear's `apply` on `DraftGemm` (2-D BF16 inputs without
    bias; anything else goes to the original method). A subclass of
    `UnquantizedLinearMethod` so type checks on the layer still hold."""

    def __init__(self, layer, inner):  # super().__init__ not needed
        self.inner = inner
        self.gemm = DraftGemm(layer.weight)

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def apply(self, layer, x, bias=None):
        if (
            bias is not None
            or x.dim() != 2
            or x.dtype != torch.bfloat16
            or layer.weight is not self.gemm.weight
        ):
            return self.inner.apply(layer, x, bias)
        out = torch.empty(x.shape[0], self.gemm.n, dtype=x.dtype, device=x.device)
        return self.gemm.run(x.contiguous(), out)


def install(module) -> int:
    """Route every unquantized BF16 linear under `module` through DraftGemm
    (not `kv_b_proj`: the decode runs its absorbed W_UK / W_UV instead)."""
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    count = 0
    for name, sub in module.named_modules():
        if not isinstance(sub, LinearBase) or name.endswith("kv_b_proj"):
            continue
        method = getattr(sub, "quant_method", None)
        if type(method) is not UnquantizedLinearMethod:
            continue
        weight = getattr(sub, "weight", None)
        if weight is None or weight.dtype != torch.bfloat16 or weight.dim() != 2:
            continue
        sub.quant_method = DraftLinearMethod(sub, method)
        count += 1
    return count


def draft_logits(model, hidden_states):
    """`model.compute_draft_logits`: the lm_head shard (the target's lm_head weight,
    shared; on `DraftGemm` for a plain BF16 head, else its own method), then the
    vocab gather on the drafter's mailbox (`comm.DrafterComm.gather_logits`; vLLM's
    all-gather for a shape it does not serve). BF16 logits only (an FP32 head keeps
    `compute_logits`)."""
    lp, head = model.logits_processor, model.lm_head
    cached = getattr(model, "_draft_lm_head_gemm", None)
    if cached is None or cached[0] is not head.weight:
        gemm = None
        if (
            type(head.quant_method).__name__ == "UnquantizedEmbeddingMethod"
            and head.weight.dtype == torch.bfloat16
            and head.weight.dim() == 2
            and lp.head_dtype in (None, torch.bfloat16)
            and not lp.logits_as_input
        ):
            gemm = DraftGemm(head.weight)
        cached = (head.weight, gemm)
        model._draft_lm_head_gemm = cached
    gemm, comm = cached[1], getattr(model, "_drafter_comm", None)
    if hidden_states.dim() != 2 or lp.logits_as_input:
        return model.compute_logits(hidden_states)
    if gemm is None and (
        comm is None
        or head.weight.dtype != torch.bfloat16
        or lp.head_dtype not in (None, torch.bfloat16)
    ):
        return model.compute_logits(hidden_states)
    x = hidden_states.contiguous()
    if gemm is not None:
        logits = torch.empty(x.shape[0], gemm.n, dtype=x.dtype, device=x.device)
        gemm.run(x, logits)
    else:
        logits = lp._apply_head(head, x, None)
    if head.tp_size > 1:
        gathered = None if comm is None else comm.gather_logits(logits)
        logits = lp._gather_logits(logits) if gathered is None else gathered
    logits = logits[..., : lp.org_vocab_size]
    if lp.soft_cap is not None:
        logits = torch.tanh(logits / lp.soft_cap) * lp.soft_cap
    if lp.scale != 1.0:
        logits *= lp.scale
    return logits
