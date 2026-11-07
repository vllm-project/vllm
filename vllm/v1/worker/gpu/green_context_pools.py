# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA green-context SM pools for SubGPU pipelining.

Kernels launched on a green-context stream run only on that context's SMs, so
the attention pool and the rest-of-layer pool partition the GPU. Both pools are
split out of one container green context, by whichever count is a multiple of
8 (the split granularity); the other pool gets the remainder.

The driver API is used directly because the two pools must be disjoint:
`torch.cuda.green_contexts.GreenContext(num_sms=...)` (torch 2.14) creates
independent contexts that may share SMs, and `SMPartition.split` is only in
later torch releases.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from cuda.bindings import driver as _cu
except ImportError:
    try:
        from cuda import cuda as _cu
    except ImportError:
        _cu = None


def _check(ret):
    """Unpack the (err, *outputs) convention of cuda.bindings.driver."""
    if isinstance(ret, tuple):
        err, outs = ret[0], ret[1:]
    else:
        err, outs = ret, ()
    if err != _cu.CUresult.CUDA_SUCCESS:
        try:
            _, name = _cu.cuGetErrorName(err)
            name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            name = f"CUresult({int(err)})"
        raise RuntimeError(f"CUDA driver error: {name}")
    if not outs:
        return None
    return outs[0] if len(outs) == 1 else outs


def _sm_of(res):
    sm = getattr(res, "sm", None)
    return sm if sm is not None else res.data.sm


def _split(res, min_count: int, flags: int = 0):
    groups, nb, rem = _check(_cu.cuDevSmResourceSplitByCount(1, res, flags, min_count))
    valid = list(groups[:nb]) if groups is not None else []
    rem_count = int(_sm_of(rem).smCount) if rem is not None else 0
    return valid, (rem if rem_count > 0 else None)


class GreenContextPools:
    """An attention pool with one stream and a rest pool with `num_rest_streams`
    streams. The streams are `torch.cuda.ExternalStream`s, which do not own
    their handles, so this object must outlive them."""

    def __init__(
        self,
        device_index: int,
        attn_sms: int,
        rest_sms: int,
        num_rest_streams: int = 2,
    ) -> None:
        if _cu is None:
            raise RuntimeError("SubGPU pipelining requires cuda-python.")
        if rest_sms % 8 == 0:
            split_count, split_is_attn = rest_sms, False
        elif attn_sms % 8 == 0:
            split_count, split_is_attn = attn_sms, True
        else:
            raise ValueError(
                f"Neither attn_sms={attn_sms} nor rest_sms={rest_sms} is a "
                "multiple of 8."
            )
        self._gctxs: list = []
        self._raw_streams: list = []
        _check(_cu.cuInit(0))
        dev = _check(_cu.cuDeviceGet(device_index))
        sm_type = _cu.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        flags = _cu.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM

        def make_gctx(res):
            desc = _check(_cu.cuDevResourceGenerateDesc([res], 1))
            gctx = _check(_cu.cuGreenCtxCreate(desc, dev, flags))
            self._gctxs.append(gctx)
            return gctx

        def sm_count(gctx) -> int:
            res = _check(_cu.cuGreenCtxGetDevResource(gctx, sm_type))
            return int(_sm_of(res).smCount)

        def make_stream(gctx):
            nonblocking = _cu.CUstream_flags.CU_STREAM_NON_BLOCKING
            raw = _check(_cu.cuGreenCtxStreamCreate(gctx, nonblocking, 0))
            self._raw_streams.append(raw)
            return torch.cuda.ExternalStream(int(raw), device=f"cuda:{device_index}")

        try:
            dev_res = _check(_cu.cuDeviceGetDevResource(dev, sm_type))
            container, _ = _split(dev_res, attn_sms + rest_sms)
            if not container:
                raise RuntimeError("container green-context split failed")
            parent = _check(
                _cu.cuGreenCtxGetDevResource(make_gctx(container[0]), sm_type)
            )
            groups, rem = _split(parent, split_count)
            if not groups or rem is None:
                raise RuntimeError(f"split({split_count}) gave no group + remainder")
            res_attn, res_rest = (groups[0], rem) if split_is_attn else (rem, groups[0])
            self.attn_gctx = make_gctx(res_attn)
            self.rest_gctx = make_gctx(res_rest)
            self.attn_sms = sm_count(self.attn_gctx)
            self.rest_sms = sm_count(self.rest_gctx)
            self.attn_stream = make_stream(self.attn_gctx)
            self.rest_streams = [
                make_stream(self.rest_gctx) for _ in range(num_rest_streams)
            ]
        except Exception:
            self._destroy()
            raise
        logger.info(
            "SubGPU pipelining: attention pool %d SMs, rest pool %d SMs, "
            "%d rest stream(s).",
            self.attn_sms,
            self.rest_sms,
            len(self.rest_streams),
        )

    def _destroy(self) -> None:
        for stream in self._raw_streams:
            _cu.cuStreamDestroy(stream)
        self._raw_streams = []
        for gctx in reversed(self._gctxs):
            _cu.cuGreenCtxDestroy(gctx)
        self._gctxs = []

    def close(self) -> None:
        torch.cuda.synchronize()
        self._destroy()
