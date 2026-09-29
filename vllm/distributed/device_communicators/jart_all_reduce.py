# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-reduce for gfx11 at TP4 with the LL protocol (csrc/rocm/jart_ar.cu).

Reduce-scatter + all-gather where every 64-bit packet carries its data and
its flag, so there are no barriers: a rank knows a peer's chunk arrived when
the flag in the same packet matches the call. Measured on a local TP4 box with
MTP k=3: decode step -13% with one request, -8% with two, -6% with three.

It is only used for all-reduces being captured into a CUDA graph: the flags
advance per call, and capture replays the same sequence on every rank, which
eager execution does not guarantee. Everything else takes the usual path.

``VLLM_JART_AR_FUSE`` also fuses all-reduce + residual add + Gemma RMSNorm
into one kernel through an Inductor pass (``JartAllReduceRMSNormPass``).
"""

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Messages up to 128 rows x 8192 halves (the capture sizes this is built for).
_MAX_ROWS, _MAX_COLS = 128, 8192


class JartAllReduce:
    def __init__(self, cpu_group: ProcessGroup, device: torch.device) -> None:
        import vllm._rocm_C  # noqa: F401  registers torch.ops._jart_ar

        self.disabled = True
        self.device = device
        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        if self.world != 4:
            raise ValueError("jart all-reduce supports TP4 only")
        self.ops = torch.ops._jart_ar
        self.algo = envs.VLLM_JART_AR_ALGO
        self.max_numel = envs.VLLM_JART_AR_MAX_NUMEL
        with torch.cuda.device(device):
            ptr, handle = self.ops.alloc_shared()
            handles: list = [None] * self.world
            dist.all_gather_object(handles, handle, group=cpu_group)
            ptrs = [
                ptr if r == self.rank else self.ops.open_handle(handles[r])
                for r in range(self.world)
            ]
            self.ctx = self.ops.init(ptrs, self.rank)
        self.disabled = False

    def should_use(self, inp: torch.Tensor) -> bool:
        return (
            not self.disabled
            and inp.device == self.device
            and inp.dtype in (torch.float16, torch.bfloat16)
            and inp.dim() == 2
            and 0 < inp.shape[0] <= _MAX_ROWS
            and 0 < inp.shape[1] <= _MAX_COLS
            and inp.numel() % 8 == 0
            and inp.numel() < self.max_numel
            and torch.cuda.is_current_stream_capturing()
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        if not inp.is_contiguous() or inp.data_ptr() % 16:
            inp = inp.clone(memory_format=torch.contiguous_format)
        out = torch.empty_like(inp)
        self.ops.all_reduce(self.ctx, inp, out, self.algo)
        return out


def maybe_create_jart_all_reduce(
    cpu_group: ProcessGroup, device: torch.device, world_size: int
) -> JartAllReduce | None:
    """Create it on every rank or on none: a failure anywhere disables it."""
    if not envs.VLLM_JART_AR or world_size != 4:
        return None
    comm, err = None, None
    try:
        comm = JartAllReduce(cpu_group, device)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    errs: list = [None] * world_size
    dist.all_gather_object(errs, err, group=cpu_group)
    if any(errs):
        logger.warning("jart all-reduce disabled, failed on some rank: %s", errs)
        return None
    if envs.VLLM_JART_AR_FUSE:
        _register_fused_op()
    logger.info(
        "jart all-reduce active (algo %d, < %d elements, graph capture only)",
        comm.algo,
        comm.max_numel,
    )
    return comm


def _register_fused_op() -> None:
    if hasattr(torch.ops, "jart") and hasattr(torch.ops.jart, "ar_add_rms"):
        return

    @torch.library.custom_op("jart::ar_add_rms", mutates_args=())
    def ar_add_rms(
        x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """all_reduce(x) + res, then Gemma RMSNorm with weight (w + 1).

        One kernel under graph capture; the unfused ops otherwise.
        """
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce

        comm = getattr(get_tp_group().device_communicator, "jart_ar_comm", None)
        if (
            comm is not None
            and comm.should_use(x)
            and x.is_contiguous()
            and res.is_contiguous()
        ):
            out, res_out = torch.empty_like(x), torch.empty_like(x)
            comm.ops.ar_add_rms(comm.ctx, x, res, w, eps, out, res_out)
            return out, res_out
        xf = tensor_model_parallel_all_reduce(x).float() + res.float()
        res_out = xf.to(x.dtype)
        var = xf.pow(2).mean(-1, keepdim=True)
        out = (xf * torch.rsqrt(var + eps)) * (w.float() + 1.0)
        return out.to(x.dtype), res_out

    @ar_add_rms.register_fake
    def _(x, res, w, eps):
        return torch.empty_like(x), torch.empty_like(x)


def make_jart_fusion_pass(config):
    """Inductor pass: all_reduce + fused_add_rms_norm(w + 1) -> jart::ar_add_rms."""
    import torch._inductor.pattern_matcher as pm

    import vllm.ir.ops
    from vllm.compilation.passes.vllm_inductor_pass import (
        VllmFusionPatternMatcherPass,
        VllmPatternReplacement,
    )
    from vllm.distributed import tensor_model_parallel_all_reduce

    _register_fused_op()

    class JartAllReduceRMSNorm(VllmPatternReplacement):
        def __init__(self, eps: float, dtype: torch.dtype) -> None:
            self.eps, self.dtype = eps, dtype

        def get_inputs(self):
            return [
                self.empty(5, 16, dtype=self.dtype),
                self.empty(5, 16, dtype=self.dtype),
                self.empty(16, dtype=self.dtype),
            ]

        @property
        def pattern(self):
            def _pattern(res, x, w):
                ar = tensor_model_parallel_all_reduce(x)
                return vllm.ir.ops.fused_add_rms_norm(
                    ar, res, w.float() + 1.0, self.eps
                )

            return _pattern

        @property
        def replacement(self):
            def _replacement(res, x, w):
                out = torch.ops.jart.ar_add_rms(x, res, w, self.eps)
                return out[0], out[1]

            return _replacement

    class JartAllReduceRMSNormLast(JartAllReduceRMSNorm):
        """Same, with the residual output dead (last layer)."""

        @property
        def pattern(self):
            p = super().pattern
            return lambda res, x, w: p(res, x, w)[0]

        @property
        def replacement(self):
            r = super().replacement
            return lambda res, x, w: r(res, x, w)[0]

    class JartAllReduceRMSNormPass(VllmFusionPatternMatcherPass):
        def __init__(self, config) -> None:
            super().__init__(config, "jart_ar_rms_fusion_pass")
            for eps in (1e-6, 1e-5):
                self.register(JartAllReduceRMSNorm(eps, self.model_dtype))
                self.register(JartAllReduceRMSNormLast(eps, self.model_dtype))
                pm._seen_patterns.clear()

        def is_applicable_for_range(self, compile_range) -> bool:
            # The op decides at run time (capture + size).
            return True

    return JartAllReduceRMSNormPass(config)
