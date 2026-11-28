# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completed-block causal MXFP8 attention through cuDNN Frontend.

`sdpa_mxfp8(causal_exclude_own_block=32)` (cuDNN Frontend 1.33) restricts
query `i` to keys `j < 32 * floor(i / 32)`, so every 32-token V scale it reads
covers only visible tokens. Rows 0-31 read no key and return O = 0 and
LSE = -inf; the boundary merge discards them.
"""

import math
from functools import cache

import cudnn
import torch

BLOCK = 32
_D = 128


@cache
def _handle(device_index):
    return cudnn.create_handle()


_streams: dict[int, int] = {}


def _bound_handle(device):
    """The device's cuDNN handle, set to the current PyTorch stream."""
    index = device.index
    if index is None:
        index = torch.accelerator.current_device_index()
    handle = _handle(index)
    stream = torch.cuda.current_stream(device).cuda_stream
    # cudnnSetStream queries the driver; only rebind when the stream changes.
    if _streams.get(index) != stream:
        cudnn.set_stream(handle=handle, stream=stream)
        _streams[index] = stream
    return handle


class _Plan:
    """One built `sdpa_mxfp8` graph for dense BSHD Q/K/V of length `n`."""

    def __init__(self, b, hq, hk, n, device):
        e4m3 = cudnn.data_type.FP8_E4M3
        g = cudnn.pygraph(
            io_data_type=e4m3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=_bound_handle(device),
        )

        def payload(h, name):
            return g.tensor(
                dim=[b, h, n, _D],
                stride=[n * h * _D, _D, h * _D, 1],
                data_type=e4m3,
                name=name,
            )

        def scales(dims, name):
            # F8_128x4: the byte order `boundary.prepare` writes.
            return g.tensor(
                dim=list(dims),
                stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
                data_type=cudnn.data_type.FP8_E8M0,
                reordering_type=cudnn.tensor_reordering.F8_128x4,
                name=name,
            )

        self.q, self.k, self.v = (
            payload(hq, "q"),
            payload(hk, "k"),
            payload(hk, "v"),
        )
        self.sq = scales((b, hq, n, 4), "sf_q")
        self.sk = scales((b, hk, n, 4), "sf_k")
        self.sv = scales((b, hk, n // BLOCK, _D), "sf_v")
        self.o, self.lse, _ = g.sdpa_mxfp8(
            q=self.q,
            k=self.k,
            v=self.v,
            descale_q=self.sq,
            descale_k=self.sk,
            descale_v=self.sv,
            attn_scale=1.0 / math.sqrt(_D),
            generate_stats=True,
            use_causal_mask=True,
            causal_exclude_own_block=BLOCK,
        )
        self.o.set_output(True).set_dim([b, hq, n, _D]).set_stride(
            [n * hq * _D, _D, hq * _D, 1]
        ).set_data_type(cudnn.data_type.BFLOAT16)
        self.lse.set_output(True).set_dim([b, hq, n, 1]).set_stride(
            [hq * n, n, 1, 1]
        ).set_data_type(cudnn.data_type.FLOAT)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        self.graph = g
        self.workspace_bytes = max(g.get_workspace_size(), 1)
        self.shape = (b, hq, hk, n)

    def execute(self, q8, k8, v8, sq, sk, sv, out, lse):
        b, hq, hk, n = self.shape
        handle = _bound_handle(q8.device)
        workspace = torch.empty(
            self.workspace_bytes, device=q8.device, dtype=torch.uint8
        )
        self.graph.execute(
            {
                self.q: q8,
                self.k: k8,
                self.v: v8,
                self.sq: sq.view(b, hq, n, 4),
                self.sk: sk.view(b, hk, n, 4),
                self.sv: sv.view(b, hk, n // BLOCK, _D),
                self.o: out,
                self.lse: lse.unsqueeze(-1),
            },
            workspace,
            handle=handle,
        )


@cache
def plan(b, hq, hk, n, device):
    return _Plan(b, hq, hk, n, device)


def execute(q8, k8, v8, sq, sk, sv):
    """Completed-block attention on prepared operands: normalized BF16 partial
    in BSHD storage and its natural-log LSE `[B, H, N]`."""
    b, hq, n, _ = q8.shape
    out = torch.empty((b, n, hq, _D), device=q8.device, dtype=torch.bfloat16)
    out = out.transpose(1, 2)
    lse = torch.empty((b, hq, n), device=q8.device, dtype=torch.float32)
    plan(b, hq, k8.shape[1], n, q8.device).execute(q8, k8, v8, sq, sk, sv, out, lse)
    return out, lse
