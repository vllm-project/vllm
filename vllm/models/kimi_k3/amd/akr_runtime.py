# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide state of the kdamoe amd_kernels routes: the 8-rank communicators
(created once after the weights are loaded: a TP collective), the capture-stream
pre-warm, and the helpers the decoder layers use.

The kernels with persistent per-(device, stream) workspaces (router slabs, experts
slots, down-projection partials) key them by stream, and vLLM captures its CUDA graphs
on a dedicated stream, so the first call on that stream would allocate (and capture a
memset of) a fresh workspace inside the capture. `install_capture_hook` therefore runs
one eager call of each kernel on the capture stream when `graph_capture` is entered.
"""

import contextlib
import time
import types
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

attnres_comm: Any = None  # attnres_rmsnorm_common.attnres_comm
latent_comm: Any = None  # latent_moe_common.latent_tail_comm
_prewarm: list = []  # callables run once per capture stream
_warmed_streams: set[int] = set()
_hooked = False


def init_comms(
    attnres: bool, latent: bool, *, hidden: int, up_rows: int = 0, latent_dim: int = 0
) -> None:
    """Create the TP communicators (collective: every rank, same order) for the model's
    `hidden` size and the latent tail's up-projection shard [up_rows, latent_dim]; a
    communicator the entry has no kernel for stays None (warned once)."""
    global attnres_comm, latent_comm
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.kimi_k3.amd import akr_config

    tp = get_tp_group()
    rank, group = tp.rank_in_group, tp.cpu_group
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    t0 = time.time()
    if attnres and attnres_comm is None:
        from amd_kernels.norm import attnres_rmsnorm_common as ar

        if not akr_config.unsupported_reason(
            "norm", ar.attnres_comm_unsupported_reason(rank, group, dev, hidden=hidden)
        ):
            attnres_comm = ar.attnres_comm(rank, group, dev, hidden=hidden)
    if latent and latent_comm is None:
        from amd_kernels.moe import latent_moe_common as lm

        kw = dict(up_rows=up_rows, latent=latent_dim)
        if not akr_config.unsupported_reason(
            "moe", lm.latent_tail_comm_unsupported_reason(rank, group, dev, **kw)
        ):
            latent_comm = lm.latent_tail_comm(rank, group, dev, **kw)
    logger.info("amd_kernels kdamoe communicators ready (%.1f s)", time.time() - t0)


def register_prewarm(fn) -> None:
    _prewarm.append(fn)


def install_capture_hook() -> None:
    """Wrap the TP group's `graph_capture` (as the MiniMax bridge does) so the
    stream-keyed workspaces of the kernels exist before the capture starts."""
    global _hooked
    if _hooked:
        return
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    original = tp.graph_capture

    @contextlib.contextmanager
    def graph_capture(self, graph_capture_context=None):
        with original(graph_capture_context) as ctx:
            handle = torch.cuda.current_stream().cuda_stream
            if handle not in _warmed_streams:
                _warmed_streams.add(handle)
                t0 = time.time()
                for fn in _prewarm:
                    fn()
                torch.accelerator.synchronize()
                logger.info(
                    "amd_kernels kdamoe: capture-stream workspaces ready (%.1f s)",
                    time.time() - t0,
                )
            yield ctx

    tp.graph_capture = types.MethodType(graph_capture, tp)  # type: ignore[method-assign]
    _hooked = True
