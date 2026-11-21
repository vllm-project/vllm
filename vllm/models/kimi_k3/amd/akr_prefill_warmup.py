# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start-up warm-up of the prefill routes (compile every variant before serving:
`compile_watch` arms afterwards).

`run(vllm_config)` is called once per worker after the model is loaded and before
`compile_watch.arm` (gpu_worker.py); it touches only scratch buffers it allocates (the
KDA state / conv rows it writes are its own dummy rows). Failures are logged, never
raised: an unwarmed variant compiles on its first real use (as without this call).
"""

import torch

from vllm.logger import init_logger

from . import akr_config
from . import akr_indexer_prefill as akr_idx

logger = init_logger(__name__)


def warm_indexer(device) -> None:
    dev = device
    if akr_idx.score_on():
        from amd_kernels.indexer.mxfp4_indexer_common import mxfp4_prefill_score

        rows, pages = 64, 64
        mxfp4_prefill_score(
            torch.zeros(rows, 32, 64, dtype=torch.uint8, device=dev),
            torch.zeros(rows, 1, 4, 16, 4, dtype=torch.uint8, device=dev),
            torch.zeros(rows, 32, dtype=torch.bfloat16, device=dev),
            torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8, device=dev),
            torch.zeros(pages, 1, 4, 64, dtype=torch.uint8, device=dev),
            torch.zeros(rows, dtype=torch.int32, device=dev),
            torch.full((rows,), pages * 64, dtype=torch.int32, device=dev),
            torch.empty(rows, pages * 64, dtype=torch.float32, device=dev),
            1.0,
        )
    else:
        # aiter scorer: one compiled kernel per block-table width bucket (4096 / 8192 /
        # 16384 pages).
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
            flydsl_pa_mqa_logits_fp4_prefill,
        )

        rows = 16
        widths = (4096, 8192, 16384) if akr_idx.table_bucket_on() else ()
        for tw in widths:
            pages = tw - 4
            width = pages * 64
            payload = torch.zeros(pages, 1, 4, 64, 16, dtype=torch.uint8, device=dev)
            scales = torch.zeros(pages, 1, 4, 64, dtype=torch.uint8, device=dev)
            st = torch.zeros(rows, dtype=torch.int32, device=dev)
            en = torch.full((rows,), width, dtype=torch.int32, device=dev)
            batches = torch.zeros(rows, dtype=torch.int32, device=dev)
            _, sched, nct = compute_prefill_schedule(
                batches, st, en, 256, max(4096, rows), width
            )
            flydsl_pa_mqa_logits_fp4_prefill(
                torch.zeros(rows, 32, 64, dtype=torch.uint8, device=dev),
                torch.zeros(rows, 1, 4, 16, 4, dtype=torch.uint8, device=dev),
                payload,
                scales,
                akr_idx.identity_page_table(torch.device(dev), tw),
                torch.zeros(rows, 32, dtype=torch.bfloat16, device=dev),
                batches,
                st,
                en,
                width,
                weight_scale=1.0,
                cta_info=sched,
                n_ctas=nct,
                out=torch.empty(rows, width, dtype=torch.float32, device=dev),
            )
    if akr_idx.topk_on():
        from amd_kernels.indexer.mxfp4_indexer_common import mxfp4_prefill_topk

        rows, width = 16, 8192
        mxfp4_prefill_topk(
            torch.zeros(rows, width, dtype=torch.float32, device=dev),
            torch.zeros(rows, dtype=torch.int32, device=dev),
            torch.full((rows,), width, dtype=torch.int32, device=dev),
            torch.empty(rows, akr_idx.TOPK, dtype=torch.int32, device=dev),
        )
    torch.accelerator.synchronize()


def warm_kda(layer) -> None:
    from . import akr_kda_prefill as ak

    if not (akr_config.on("kda") and ak._env_on()):
        return
    from amd_kernels.attention.kda_common import kda_prefill, kda_prefill_shapes

    dev = layer.A_log.device
    h, d = layer.local_num_heads, layer.head_dim
    if (h, d) not in kda_prefill_shapes():
        return
    hd = h * d
    n = 64
    rows = 8
    row = 6288
    proj = torch.zeros(n, row, dtype=torch.bfloat16, device=dev)
    conv_state = torch.zeros(
        rows, 3, 3 * hd, dtype=torch.bfloat16, device=dev
    ).transpose(1, 2)
    rec = torch.zeros(rows, h, d, d, dtype=torch.float32, device=dev)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    ci = torch.tensor([[0, 0]], dtype=torch.int32, device=dev)
    co = torch.tensor([0, 1], dtype=torch.int64, device=dev)
    sidx = torch.tensor([1], dtype=torch.int32, device=dev)
    has = torch.zeros(1, dtype=torch.bool, device=dev)
    out = torch.empty(n, h, d, dtype=torch.bfloat16, device=dev)
    for ckpt in (False, True):
        kw = {}
        if ckpt:
            kw = dict(
                checkpoint_state=rec,
                checkpoint_offsets=torch.zeros(1, dtype=torch.int32, device=dev),
                checkpoint_rows=torch.tensor([2], dtype=torch.int32, device=dev),
            )
        kda_prefill(
            proj[:, : 3 * hd],
            torch.zeros(n, h, d, dtype=torch.bfloat16, device=dev),
            proj[:, 6272 : 6272 + h],
            proj[:, 3 * hd : 4 * hd].view(n, h, d),
            layer.A_log,
            layer.dt_bias,
            ak._conv_weight(layer),
            conv_state,
            rec,
            sidx,
            has,
            cu,
            ci,
            co,
            layer.o_norm.weight,
            layer.o_norm.eps,
            out,
            layer.gate_lower_bound,
            chunk_counts=[1],
            **kw,
        )
    torch.accelerator.synchronize()


def run(vllm_config) -> None:
    if not akr_config.routes():
        return
    try:
        from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention

        layers = [
            m
            for m in vllm_config.compilation_config.static_forward_context.values()
            if isinstance(m, KimiK3DeltaAttention)
        ]
        if not layers:
            return  # not a Kimi-K3 ROCm model
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        warm_indexer(dev)
        warm_kda(layers[0])
        logger.info("amd_kernels prefill routes warmed up")
    except Exception:  # noqa: BLE001
        logger.warning("amd_kernels prefill warm-up failed", exc_info=True)
