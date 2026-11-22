# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-load hook of the DeepSeek-V4.1 amd_kernels routes: weight layouts,
static workspaces and kernel warm-ups, all before CUDA graph capture.

Each route module has `post_load(model)` and runs for the target and the DSpark
draft model. A module that raises takes its routes out (`akr_config.fail`), so
the stock kernels run.
"""

import importlib

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41.amd import akr_config

logger = init_logger(__name__)

# (module, routes it owns), in dependency order
_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("akr_moe", ("router", "shared_expert", "draft_router")),
    (
        "akr_attn",
        ("rope_insert", "norm_dual", "compressor", "kv_score", "dense_gemm", "wq_b"),
    ),
    ("akr_indexer", ("indexer",)),
    ("akr_mla", ("sparse_mla", "wo_a", "wo_b")),
    ("akr_experts", ("fused_ep8",)),
    ("akr_mhc", ("mhc_seam", "mhc_tail", "mhc_wo_b", "mhc_ffn")),
    ("akr_draft", ("lm_head", "main_proj")),
    ("akr_sample", ("markov",)),
)


# whether a DeepSeek-V4.1 model of this process ran the post-load hook
loaded = False


def is_draft(model: torch.nn.Module) -> bool:
    """The DSpark draft model. `akr_config.fail` is process-wide, so only the
    target model may take a route out for lacking eligible layers."""
    return hasattr(getattr(model, "model", None), "markov_head")


def run_post_load(model: torch.nn.Module) -> None:
    global loaded
    if not akr_config.routes():
        return
    loaded = True
    # allocate fp32 unless a kernel asks for another dtype explicitly
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        for name, owned in _MODULES:
            if not any(akr_config.on(r) for r in owned):
                continue
            try:
                module = importlib.import_module(f"vllm.models.deepseek_v41.amd.{name}")
                module.post_load(model)
            except Exception as e:
                logger.exception("amd_kernels %s post_load failed", name)
                for r in owned:
                    akr_config.fail(r, f"{name}.post_load: {type(e).__name__}: {e}")
    finally:
        torch.set_default_dtype(prev)
