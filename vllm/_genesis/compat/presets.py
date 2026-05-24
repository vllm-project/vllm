# SPDX-License-Identifier: Apache-2.0
"""Genesis preset matrix — curated `(gpu_class × workload) → bundle`.

Distinct from `recipes.py` (which manages USER-saved recipes). Presets
are CURATED bundles maintained by the Genesis project + downstream
consumers (club-3090, etc) — the equivalent of `apt install <preset>`
for vLLM patch configurations.

A preset bundles:
  - Genesis env vars (GENESIS_ENABLE_PXX, threshold knobs)
  - vLLM serve flags (--max-model-len, --gpu-memory-utilization, etc)
  - Recommended model + quantization
  - Tensor parallel hint
  - System env (PYTORCH_CUDA_ALLOC_CONF, NCCL_*, etc)
  - Verified-on rigs (cross-rig provenance — empirical evidence, not theory)
  - Expected TPS reference (so operators know if their bench number is
    reasonable or off)
  - Notes (gotchas, conflicts, when NOT to pick this preset)

Usage:
    from vllm._genesis.compat.presets import (
        list_presets,
        get_preset,
        match_preset,
    )

    # List all curated presets
    presets = list_presets()

    # Get specific preset by key
    p = get_preset("a5000-2x-balanced")

    # Auto-match by GPU + workload
    p = match_preset(gpu_class="rtx_3090", n_gpus=1, workload="long_context")

    # Render to launch script (env block + vllm serve command)
    print(p.to_launch_script(model_path="/models/qwen3.6-27b"))

CLI: `python3 -m vllm._genesis.compat.presets list`
     `python3 -m vllm._genesis.compat.presets show a5000-2x-balanced`
     `python3 -m vllm._genesis.compat.presets match --gpu rtx_a5000 --n-gpus 2 --workload balanced`

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional


# Workload taxonomy. The 4 categories cover ~95% of consumer/prosumer use.
WORKLOADS = {
    "long_context": (
        "Single long prompt (>50K), low concurrency, e.g. RAG over a "
        "codebase, doc summarization, structured CoT. Optimizes for "
        "context ceiling + safe Cliff 2 handling."
    ),
    "high_throughput": (
        "Many short prompts in parallel, max TPS, e.g. batch inference, "
        "synthetic-data generation. Optimizes for tokens/sec at the "
        "expense of per-request latency."
    ),
    "tool_agent": (
        "IDE coding agents (Cline / Claude Code / OpenCode / Roo), "
        "function calling, multi-tool. Optimizes for clean tool-call "
        "parse rate + reasoning quality."
    ),
    "balanced": (
        "Mixed workload — chat, occasional long context, occasional "
        "tools. Default-safe preset when workload isn't known up front."
    ),
}


@dataclass
class Preset:
    """A curated launch bundle for a (gpu × workload) combination."""

    key: str
    """Stable identifier, e.g. 'a5000-2x-balanced'."""

    title: str
    """Human-readable name."""

    description: str
    """1-2 sentences explaining when to pick this preset."""

    workload: str
    """One of WORKLOADS keys."""

    gpu_match_keys: list[str]
    """gpu_profile.GPU_SPECS match keys this preset is verified on,
    e.g. ['rtx a5000', 'rtx 3090']."""

    n_gpus: int
    """Tensor parallel size."""

    model_recommended: str
    """Recommended HF model ID, e.g. 'Qwen/Qwen3-Next-80B-A3B-FP8'."""

    quantization: Optional[str] = None
    """vLLM --quantization arg, e.g. 'auto_round', 'fp8', or None."""

    kv_cache_dtype: Optional[str] = None
    """vLLM --kv-cache-dtype arg, e.g. 'turboquant_k8v4', 'fp8', None."""

    max_model_len: int = 32768
    """vLLM --max-model-len."""

    gpu_memory_utilization: float = 0.90
    """vLLM --gpu-memory-utilization."""

    max_num_batched_tokens: int = 4096
    """vLLM --max-num-batched-tokens."""

    speculative_method: Optional[str] = None
    """One of 'mtp', 'eagle', 'ngram', 'dflash', None."""

    num_speculative_tokens: Optional[int] = None
    """K for spec-decode."""

    genesis_env: dict[str, str] = field(default_factory=dict)
    """GENESIS_ENABLE_* + threshold knobs."""

    vllm_extra_args: list[str] = field(default_factory=list)
    """Extra vllm serve flags beyond the standard ones above."""

    system_env: dict[str, str] = field(default_factory=dict)
    """System env vars (PYTORCH_CUDA_ALLOC_CONF, NCCL_*, VLLM_*)."""

    verified_on: list[str] = field(default_factory=list)
    """Cross-rig provenance — list of '<owner>/<rig>: <metric>' entries.
    Empirical evidence the preset works (not theory)."""

    expected_tps_ref: Optional[str] = None
    """Reference TPS so operators know roughly what to expect.
    Format: '<n> tok/s @ <ctx> tokens, CV <pct>%' or freeform."""

    notes: list[str] = field(default_factory=list)
    """Gotchas, conflicts, when NOT to pick this preset. Each entry
    starts with a tag like '⚠' / 'ℹ' / 'WHEN NOT'."""

    def to_dict(self) -> dict:
        """JSON-serializable dict view."""
        return asdict(self)

    def to_launch_script(
        self,
        model_path: Optional[str] = None,
        port: int = 8000,
        served_model_name: Optional[str] = None,
    ) -> str:
        """Render preset as a runnable bash launch script.

        Args:
          model_path: override model path (else uses self.model_recommended)
          port: HTTP serve port
          served_model_name: vLLM --served-model-name (else derived
            from model path basename)
        """
        model = model_path or self.model_recommended
        name = served_model_name or model.split("/")[-1]

        lines = [
            "#!/usr/bin/env bash",
            "# Generated by Genesis preset:",
            f"#   key:         {self.key}",
            f"#   title:       {self.title}",
            f"#   workload:    {self.workload}",
            f"#   gpu:         {', '.join(self.gpu_match_keys)} × {self.n_gpus}",
        ]
        if self.expected_tps_ref:
            lines.append(f"#   reference:   {self.expected_tps_ref}")
        if self.verified_on:
            lines.append("#   verified-on:")
            for v in self.verified_on:
                lines.append(f"#     - {v}")
        if self.notes:
            lines.append("#   notes:")
            for n in self.notes:
                lines.append(f"#     {n}")
        lines.extend([
            "",
            "set -euo pipefail",
            "",
        ])

        # System env block
        if self.system_env:
            lines.append("# ─── System env ───────────────────────────────────")
            for k, v in self.system_env.items():
                lines.append(f"export {k}={_shell_quote(v)}")
            lines.append("")

        # Genesis env block
        if self.genesis_env:
            lines.append("# ─── Genesis env (curated for this preset) ────────")
            for k, v in self.genesis_env.items():
                lines.append(f"export {k}={_shell_quote(v)}")
            lines.append("")

        # vllm serve command
        lines.append("# ─── vllm serve ───────────────────────────────────")
        lines.append("exec vllm serve \\")
        lines.append(f"  {_shell_quote(model)} \\")
        lines.append(f"  --served-model-name {_shell_quote(name)} \\")
        lines.append(f"  --port {port} \\")
        lines.append(f"  --tensor-parallel-size {self.n_gpus} \\")
        lines.append(f"  --max-model-len {self.max_model_len} \\")
        lines.append(f"  --max-num-batched-tokens {self.max_num_batched_tokens} \\")
        lines.append(
            f"  --gpu-memory-utilization {self.gpu_memory_utilization} \\"
        )
        if self.quantization:
            lines.append(f"  --quantization {_shell_quote(self.quantization)} \\")
        if self.kv_cache_dtype:
            lines.append(f"  --kv-cache-dtype {_shell_quote(self.kv_cache_dtype)} \\")
        if self.speculative_method:
            spec_cfg = {"method": self.speculative_method}
            if self.num_speculative_tokens:
                spec_cfg["num_speculative_tokens"] = self.num_speculative_tokens
            lines.append(
                f"  --speculative-config {_shell_quote(json.dumps(spec_cfg))} \\"
            )
        for arg in self.vllm_extra_args:
            lines.append(f"  {arg} \\")
        # strip trailing backslash on the last line
        last = lines[-1]
        if last.endswith(" \\"):
            lines[-1] = last[:-2]

        return "\n".join(lines) + "\n"


def _shell_quote(s) -> str:
    """Minimal shell-safe single-quoting."""
    s = str(s)
    # If the string is already safe (alnum + - _ . / @ : =), don't quote
    safe = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./@:="
    )
    if s and all(c in safe for c in s):
        return s
    # Otherwise single-quote, escaping any internal single quotes
    return "'" + s.replace("'", "'\\''") + "'"


# ════════════════════════════════════════════════════════════════════
# CURATED PRESET MATRIX
# ════════════════════════════════════════════════════════════════════
#
# Each preset is verified-on at least one rig before listing here.
# Cross-rig provenance lives in `verified_on`. When a preset works
# elsewhere, append (don't replace) so operators see the spread.
#
# Naming convention: <gpu_class>-<n>x-<workload>
#   gpu_class: lowercase, no spaces (a5000, 3090, 4090, 5090, ...)
#   n:         number of GPUs (1, 2, 4, 8)
#   workload:  one of WORKLOADS keys (long_context, high_throughput,
#              tool_agent, balanced)
# ════════════════════════════════════════════════════════════════════

_PRESETS: dict[str, Preset] = {}


def _register(preset: Preset) -> None:
    if preset.key in _PRESETS:
        raise ValueError(f"duplicate preset key: {preset.key!r}")
    if preset.workload not in WORKLOADS:
        raise ValueError(
            f"preset {preset.key!r}: unknown workload {preset.workload!r} "
            f"(must be one of {list(WORKLOADS.keys())})"
        )
    _PRESETS[preset.key] = preset


# ─── 2× A5000 (Sander PROD reference rig) ───────────────────────────

_register(Preset(
    key="a5000-2x-balanced",
    title="2× RTX A5000 — balanced (Sander PROD reference)",
    description=(
        "Reference PROD config running 24/7 on Sander's homelab. "
        "Qwen3.6-27B-int4 + TurboQuant k8v4 KV + MTP K=3. Solid "
        "all-rounder for chat + tool calls + medium context."
    ),
    workload="balanced",
    gpu_match_keys=["rtx a5000"],
    n_gpus=2,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=131072,
    gpu_memory_utilization=0.90,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "1",
    },
    verified_on=[
        "Sandermage/2x-A5000: 104.0 tok/s @ 256t output, CV 0.5%",
    ],
    expected_tps_ref="104 tok/s @ 256t output, CV 0.5% (5-run mean)",
    notes=[
        "ℹ Cliff 2 (>50K single prompt) NOT triggered at TP=2 — "
        "state splits across ranks.",
        "ℹ Default-safe baseline. Switch to a5000-2x-tool-agent if "
        "running IDE agents.",
    ],
))

_register(Preset(
    key="a5000-2x-tool-agent",
    title="2× RTX A5000 — IDE coding agents",
    description=(
        "Same hardware as a5000-2x-balanced, but tuned for tool "
        "calling: P61 multi-tool / P62 reasoning grammar / P68/P69 "
        "long-context tool reminders / strict ngram path."
    ),
    workload="tool_agent",
    gpu_match_keys=["rtx a5000"],
    n_gpus=2,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=131072,
    gpu_memory_utilization=0.90,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL": "1",
        "GENESIS_ENABLE_P61B_STREAMING_OVERLAP": "1",
        "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING": "1",
        "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL": "1",
        "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER": "1",
        "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS": "50000",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "1",
    },
    verified_on=[
        "Sandermage/2x-A5000: 104 tok/s, tool-call clean 7/7",
    ],
    expected_tps_ref="~104 tok/s @ 256t output, tool-call clean 7/7",
    notes=[
        "ℹ Tuned for Cline / Claude Code / OpenCode prompts with "
        "5K+ char system prompts + multi-tool schemas.",
        "ℹ P68/P69 threshold 50K chars — long sys prompts handled.",
    ],
))

_register(Preset(
    key="a5000-2x-high-throughput",
    title="2× RTX A5000 — high throughput (35B-A3B FP8)",
    description=(
        "Higher-TPS configuration on Qwen3.6-35B-A3B FP8 with MTP "
        "K=3. ~184 tok/s sustained."
    ),
    workload="high_throughput",
    gpu_match_keys=["rtx a5000"],
    n_gpus=2,
    model_recommended="Qwen/Qwen3.6-35B-A3B-FP8",
    quantization=None,
    kv_cache_dtype="fp8",
    max_model_len=65536,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=8192,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN26_SPARSE_V": "1",
        "GENESIS_PN26_SPARSE_V_BLOCK_KV": "8",
        "GENESIS_PN26_SPARSE_V_NUM_WARPS": "4",
        "GENESIS_PN26_SPARSE_V_THRESHOLD": "0.01",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[
        "Sandermage/2x-A5000: 183.7 tok/s @ 256t output",
    ],
    expected_tps_ref="~184 tok/s @ 256t output",
    notes=[
        "ℹ PN26b sparse-V kernel tuned for 27B BLOCK_KV=8 — works "
        "well on 35B too.",
    ],
))


# ─── 1× RTX 3090 (single-card community rig — noonghunna et al) ─────

_register(Preset(
    key="3090-1x-long-context",
    title="1× RTX 3090 — long context (Cliff 2 closed)",
    description=(
        "Single 24GB card, optimized for >50K single prompts. Pairs "
        "P103 (inner FLA chunked h) + PN32 v2 (outer FLA call "
        "chunked) for full Cliff 2 coverage."
    ),
    workload="long_context",
    gpu_match_keys=["rtx 3090"],
    n_gpus=1,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=214016,
    gpu_memory_utilization=0.985,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        # Core PROD set
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
        # Cliff 2 stack — required pair for >50K on 24GB
        "GENESIS_ENABLE_P103": "1",
        "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL": "1",
        "GENESIS_PN32_GDN_CHUNK_SIZE": "8192",
        "GENESIS_PN32_GDN_CHUNK_THRESHOLD": "16384",
        "GENESIS_FLA_FWD_H_MAX_T": "16384",
        # Workspace lock companion (rare runtime path)
        "GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0",
        "PYTORCH_CUDA_ALLOC_CONF": (
            "expandable_segments:True,max_split_size_mb:512"
        ),
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "1",
    },
    verified_on=[
        "noonghunna/1x-3090: 78 tok/s narrative / 128 tok/s code "
        "(club-3090 long-text variant)",
    ],
    expected_tps_ref="~78-128 tok/s depending on workload, 214K ctx ceiling",
    notes=[
        "⚠ NOT for IDE agents — use 3090-1x-tool-agent (tools-text "
        "75K + fp8 KV instead — Cliff 1 mech B otherwise).",
        "ℹ Cliff 2 closed by P103 + PN32 v2 combo (v7.69+).",
        "ℹ Drop --enable-prefix-caching — hybrid GDN crashes with it.",
    ],
))

_register(Preset(
    key="3090-1x-tool-agent",
    title="1× RTX 3090 — IDE coding agents (75K + fp8)",
    description=(
        "Single 3090 for IDE agents (Cline / Claude Code / OpenCode). "
        "Uses fp8 KV instead of TQ k8v4 — avoids Cliff 1 mech B which "
        "fires on 5K+ char system prompts + tool schemas."
    ),
    workload="tool_agent",
    gpu_match_keys=["rtx 3090"],
    n_gpus=1,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="fp8",
    max_model_len=75000,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL": "1",
        "GENESIS_ENABLE_P61B_STREAMING_OVERLAP": "1",
        "GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING": "1",
        "GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING": "1",
        "GENESIS_ENABLE_P68_AUTO_FORCE_TOOL": "1",
        "GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER": "1",
        "GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS": "50000",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[
        "noonghunna/1x-3090: stable on Cline + 5,900-char sys prompt "
        "+ 10 tool schemas (club-3090 tools-text variant)",
    ],
    expected_tps_ref="~70-90 tok/s @ 256t output",
    notes=[
        "ℹ fp8 KV (not TQ) — necessary on TP=1 + IDE agents to avoid "
        "Cliff 1 mech B inductor-leak crash.",
    ],
))


# ─── 2× RTX 3090 (community dual-card — noonghunna 'dual') ──────────

_register(Preset(
    key="3090-2x-balanced",
    title="2× RTX 3090 — balanced (Cliff 2 split across ranks)",
    description=(
        "Dual 3090 for chat + medium-long context. TP=2 splits GDN "
        "state across ranks → Cliff 2 effectively unreachable. Same "
        "patch set as Sander's PROD A5000 setup, scaled up max_model_len."
    ),
    workload="balanced",
    gpu_match_keys=["rtx 3090"],
    n_gpus=2,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=131072,
    gpu_memory_utilization=0.90,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": (
            "expandable_segments:True,max_split_size_mb:512"
        ),
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "1",
    },
    verified_on=[
        "noonghunna/2x-3090: ~116 tok/s wall_TPS",
    ],
    expected_tps_ref="~116 tok/s wall_TPS (~12% over A5000 reference)",
    notes=[
        "ℹ TP=2 splits GDN state — Cliff 2 single-card concern doesn't "
        "apply here. P103 / PN32 still available if you push past 200K.",
    ],
))


# ─── 1× RTX 4090 (single-card prosumer — 24GB) ──────────────────────

_register(Preset(
    key="4090-1x-balanced",
    title="1× RTX 4090 — balanced (24GB compute-leaning)",
    description=(
        "Single RTX 4090. 72MB L2 enables P40 (TQ grouped decode) "
        "which is no-op on Ampere. Otherwise same patch profile as "
        "1× 3090 long-context."
    ),
    workload="balanced",
    gpu_match_keys=["rtx 4090"],
    n_gpus=1,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=131072,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P40_TQ_GROUPED_DECODE": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
        "GENESIS_ENABLE_P103": "1",
        "GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[],
    expected_tps_ref="(awaiting 4090 community data)",
    notes=[
        "ℹ P40 enabled (4090 has 72MB L2 — KV stays resident across "
        "queries → grouped-decode amortization works).",
        "ℹ Cliff 2 stack also enabled defensively for >50K prompts.",
    ],
))


# ─── 1× RTX 5090 (Blackwell consumer — 32GB) ────────────────────────

_register(Preset(
    key="5090-1x-balanced",
    title="1× RTX 5090 — balanced (32GB Blackwell)",
    description=(
        "Single RTX 5090 (Blackwell). 1792 GB/s + 88MB L2 — compute-"
        "bound regime. P40 + P67 both effective."
    ),
    workload="balanced",
    gpu_match_keys=["rtx 5090"],
    n_gpus=1,
    model_recommended="local/qwen3.6-27b-autoround-int4",
    quantization="auto_round",
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=131072,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=4096,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P4": "1",
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_P40_TQ_GROUPED_DECODE": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_P100": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
        "GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[],
    expected_tps_ref="(awaiting 5090 community data — JartX has 5090 rig)",
    notes=[
        "ℹ Compute-bound regime — Marlin / Triton kernels at full pace.",
        "ℹ P40 + P67 both enabled (large L2 + large bandwidth).",
    ],
))


# ─── R6000 Pro Blackwell 96G (enterprise — Quentin / future Sander) ─

_register(Preset(
    key="r6000-1x-balanced",
    title="1× RTX PRO 6000 Blackwell 96G — balanced",
    description=(
        "Single R6000 Pro Blackwell 96G. Enterprise card, 1792 GB/s "
        "+ 88MB L2, 96GB VRAM. Comfortable for 35B + long context."
    ),
    workload="balanced",
    gpu_match_keys=["rtx pro 6000 blackwell"],
    n_gpus=1,
    model_recommended="Qwen/Qwen3.6-35B-A3B-FP8",
    quantization=None,
    kv_cache_dtype="turboquant_k8v4",
    max_model_len=262144,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=8192,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P40_TQ_GROUPED_DECODE": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P60B_TRITON_KERNEL": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_P98": "1",
        "GENESIS_ENABLE_P99": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[],
    expected_tps_ref="(awaiting R6000 community data — Quentin-M has rig)",
    notes=[
        "ℹ 96GB VRAM — Cliff 2 unreachable, no chunked-prefill needed.",
        "ℹ Compute-bound — Marlin / sparse-V kernels at full pace.",
    ],
))


# ─── 8× RTX A4000 (community dense parallelism — JartX rig) ─────────

_register(Preset(
    key="a4000-8x-high-throughput",
    title="8× RTX A4000 — high throughput (16GB ranks × 8)",
    description=(
        "8× A4000 (16GB each, 128GB total). High parallelism cluster "
        "for serving many concurrent requests. MoE 35B-A3B FP8."
    ),
    workload="high_throughput",
    gpu_match_keys=["rtx a4000"],
    n_gpus=8,
    model_recommended="Qwen/Qwen3.6-35B-A3B-FP8",
    quantization=None,
    kv_cache_dtype="fp8",
    max_model_len=65536,
    gpu_memory_utilization=0.90,
    max_num_batched_tokens=16384,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P38B_COMPILE_SAFE": "1",
        "GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_P2P_DISABLE": "1",
    },
    verified_on=[
        "JartX/8x-A4000: validated via PR #39931 (TurboQuant hybrid)",
    ],
    expected_tps_ref="(JartX baseline — exact TPS varies by request shape)",
    notes=[
        "ℹ 8 ranks × 16GB — careful with PN12 pool sizing on small per-rank VRAM.",
    ],
))


# ─── H20 (Hopper — JartX) ───────────────────────────────────────────

_register(Preset(
    key="h20-1x-high-throughput",
    title="1× H20 — high throughput (Hopper, FlashInfer GDN)",
    description=(
        "Single H20 (Hopper, SM 9.0). FlashInfer GDN prefill kernel "
        "auto-selected. ~3.3 TB/s + 50MB L2. Compute-bound."
    ),
    workload="high_throughput",
    gpu_match_keys=["h20"],
    n_gpus=1,
    model_recommended="Qwen/Qwen3.6-35B-A3B-FP8",
    quantization=None,
    kv_cache_dtype="fp8",
    max_model_len=131072,
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=16384,
    speculative_method="mtp",
    num_speculative_tokens=3,
    genesis_env={
        "GENESIS_ENABLE_P40_TQ_GROUPED_DECODE": "1",
        "GENESIS_ENABLE_P60_GDN_NGRAM_FIX": "1",
        "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
        "GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
        "GENESIS_ENABLE_PN17_FA2_LSE_CLAMP": "1",
        "GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE": "1",
    },
    system_env={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    verified_on=[
        "JartX/H20: validated via PR #39931",
    ],
    expected_tps_ref="(JartX baseline)",
    notes=[
        "ℹ Hopper SM 9.0 — FlashInfer GDN prefill auto-selected (vs FLA Triton).",
    ],
))


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════


def list_presets() -> list[Preset]:
    """Return all curated presets."""
    return list(_PRESETS.values())


def get_preset(key: str) -> Preset:
    """Get preset by key. Raises KeyError if not found."""
    if key not in _PRESETS:
        raise KeyError(
            f"unknown preset {key!r}. "
            f"Available: {sorted(_PRESETS.keys())}"
        )
    return _PRESETS[key]


def match_preset(
    *,
    gpu_class: Optional[str] = None,
    n_gpus: int = 1,
    workload: str = "balanced",
) -> Optional[Preset]:
    """Find best-match preset for (gpu_class, n_gpus, workload).

    Args:
      gpu_class: a `gpu_profile.GPU_SPECS` match_key (e.g. "rtx a5000").
        If None, returns first preset matching n_gpus + workload.
      n_gpus: tensor parallel size.
      workload: one of WORKLOADS keys.

    Returns:
      Best matching Preset, or None if no match.

    Match priority:
      1. Exact (gpu, n_gpus, workload)
      2. Exact (gpu, n_gpus) + workload='balanced' (fallback)
      3. None (caller must handle)
    """
    if workload not in WORKLOADS:
        raise ValueError(
            f"unknown workload {workload!r} (must be one of {list(WORKLOADS.keys())})"
        )

    # Pass 1: exact match
    if gpu_class is not None:
        for p in _PRESETS.values():
            if (
                gpu_class in p.gpu_match_keys
                and p.n_gpus == n_gpus
                and p.workload == workload
            ):
                return p

        # Pass 2: same GPU + n_gpus, balanced workload as fallback
        if workload != "balanced":
            for p in _PRESETS.values():
                if (
                    gpu_class in p.gpu_match_keys
                    and p.n_gpus == n_gpus
                    and p.workload == "balanced"
                ):
                    return p
    else:
        # No GPU pinned — return first match by n_gpus + workload
        for p in _PRESETS.values():
            if p.n_gpus == n_gpus and p.workload == workload:
                return p

    return None


def auto_match() -> Optional[Preset]:
    """Detect current GPU + use 'balanced' workload as default match.

    Returns None if no CUDA / unknown GPU / no preset for this rig.
    """
    try:
        from vllm._genesis.gpu_profile import detect_current_gpu
    except ImportError:
        return None
    gpu = detect_current_gpu()
    if gpu is None or gpu.get("match_key") is None:
        return None

    # Detect n_gpus via torch.cuda.device_count()
    n_gpus = 1
    try:
        import torch

        n_gpus = max(1, torch.cuda.device_count())
    except Exception:
        pass

    return match_preset(
        gpu_class=gpu["match_key"], n_gpus=n_gpus, workload="balanced"
    )


# ════════════════════════════════════════════════════════════════════
# CLI: python3 -m vllm._genesis.compat.presets {list, show, match}
# ════════════════════════════════════════════════════════════════════


def _cmd_list(args: argparse.Namespace) -> int:
    presets = list_presets()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": p.key,
                        "title": p.title,
                        "workload": p.workload,
                        "gpu": p.gpu_match_keys,
                        "n_gpus": p.n_gpus,
                        "verified_on": p.verified_on,
                    }
                    for p in presets
                ],
                indent=2,
            )
        )
        return 0

    print(f"Genesis curated presets ({len(presets)}):\n")
    print(
        f"  {'KEY':<32}  {'GPU':<14}  {'N':<3}  {'WORKLOAD':<16}  TITLE"
    )
    print(f"  {'-' * 32}  {'-' * 14}  {'-' * 3}  {'-' * 16}  -----")
    for p in sorted(presets, key=lambda x: x.key):
        gpu = (p.gpu_match_keys[0] if p.gpu_match_keys else "?")[:14]
        print(
            f"  {p.key:<32}  {gpu:<14}  {p.n_gpus:<3}  "
            f"{p.workload:<16}  {p.title}"
        )
    print()
    print("Use:  genesis preset show <key>        # full detail")
    print("      genesis preset match --gpu <gpu> --n-gpus <n> "
          "--workload <w>")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        p = get_preset(args.key)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(p.to_dict(), indent=2, default=str))
        return 0

    if args.script:
        print(p.to_launch_script(
            model_path=args.model_path,
            port=args.port,
            served_model_name=args.served_model_name,
        ))
        return 0

    # Human-readable view
    print(f"Preset: {p.key}")
    print(f"Title:  {p.title}")
    print(f"Workload: {p.workload}  ({WORKLOADS[p.workload]})")
    print(
        f"Hardware: {', '.join(p.gpu_match_keys)} × {p.n_gpus} "
        f"(tensor-parallel)"
    )
    print(f"\n{p.description}\n")
    print(f"Model:        {p.model_recommended}")
    if p.quantization:
        print(f"Quantization: {p.quantization}")
    if p.kv_cache_dtype:
        print(f"KV cache:     {p.kv_cache_dtype}")
    print(f"Max ctx:      {p.max_model_len:,}")
    print(f"GPU mem util: {p.gpu_memory_utilization}")
    print(f"Batched tok:  {p.max_num_batched_tokens:,}")
    if p.speculative_method:
        print(
            f"Spec-decode:  {p.speculative_method} "
            f"K={p.num_speculative_tokens}"
        )

    if p.genesis_env:
        print(f"\nGenesis env  ({len(p.genesis_env)} vars):")
        for k, v in p.genesis_env.items():
            print(f"  {k}={v}")

    if p.system_env:
        print(f"\nSystem env  ({len(p.system_env)} vars):")
        for k, v in p.system_env.items():
            print(f"  {k}={v}")

    if p.expected_tps_ref:
        print(f"\nExpected: {p.expected_tps_ref}")

    if p.verified_on:
        print(f"\nVerified-on:")
        for v in p.verified_on:
            print(f"  - {v}")

    if p.notes:
        print(f"\nNotes:")
        for n in p.notes:
            print(f"  {n}")

    print(
        f"\nGenerate launch script:  genesis preset show {p.key} --script"
    )
    return 0


def _cmd_match(args: argparse.Namespace) -> int:
    p = match_preset(
        gpu_class=args.gpu, n_gpus=args.n_gpus, workload=args.workload
    )
    if p is None:
        print(
            f"error: no preset matches gpu={args.gpu!r} n_gpus={args.n_gpus} "
            f"workload={args.workload!r}",
            file=sys.stderr,
        )
        print(
            f"Use `genesis preset list` to see available combinations.",
            file=sys.stderr,
        )
        return 2

    if args.json:
        print(json.dumps(p.to_dict(), indent=2, default=str))
        return 0
    if args.script:
        print(p.to_launch_script(
            model_path=args.model_path,
            port=args.port,
            served_model_name=args.served_model_name,
        ))
        return 0

    # Default: print key + 1-line summary
    print(f"Matched: {p.key}")
    print(f"  {p.title}")
    print(f"  ({p.expected_tps_ref or 'no TPS reference yet'})")
    print(
        f"\nFull detail:  genesis preset show {p.key}"
    )
    print(
        f"Launch script: genesis preset show {p.key} --script"
    )
    return 0


def _cmd_auto(args: argparse.Namespace) -> int:
    p = auto_match()
    if p is None:
        print(
            "error: auto-match failed — no CUDA, unknown GPU, or no "
            "preset for this rig+workload combination.",
            file=sys.stderr,
        )
        print(
            "Use `genesis preset list` to see available presets, then "
            "`genesis preset show <key>` to pick one manually.",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(p.to_dict(), indent=2, default=str))
        return 0
    if args.script:
        print(p.to_launch_script(
            model_path=args.model_path,
            port=args.port,
            served_model_name=args.served_model_name,
        ))
        return 0
    print(f"Auto-matched: {p.key}")
    print(f"  {p.title}")
    print(f"  GPU detected: {p.gpu_match_keys[0]} × {p.n_gpus}")
    print(f"  ({p.expected_tps_ref or 'no TPS reference yet'})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m vllm._genesis.compat.presets",
        description=(
            "Genesis curated preset matrix — "
            "(gpu × workload) → launch bundle"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list all presets")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="show preset detail")
    p_show.add_argument("key", help="preset key, e.g. a5000-2x-balanced")
    p_show.add_argument("--json", action="store_true")
    p_show.add_argument(
        "--script",
        action="store_true",
        help="render as runnable launch script (env block + vllm serve)",
    )
    p_show.add_argument(
        "--model-path", help="override model path in script output"
    )
    p_show.add_argument("--port", type=int, default=8000)
    p_show.add_argument("--served-model-name")
    p_show.set_defaults(func=_cmd_show)

    p_match = sub.add_parser(
        "match", help="find preset by (gpu, n_gpus, workload)"
    )
    p_match.add_argument(
        "--gpu",
        help="gpu_profile match_key, e.g. 'rtx a5000', 'rtx 3090'",
    )
    p_match.add_argument("--n-gpus", type=int, default=1)
    p_match.add_argument(
        "--workload", default="balanced", choices=list(WORKLOADS.keys())
    )
    p_match.add_argument("--json", action="store_true")
    p_match.add_argument("--script", action="store_true")
    p_match.add_argument("--model-path")
    p_match.add_argument("--port", type=int, default=8000)
    p_match.add_argument("--served-model-name")
    p_match.set_defaults(func=_cmd_match)

    p_auto = sub.add_parser(
        "auto", help="auto-detect GPU + match preset (balanced workload)"
    )
    p_auto.add_argument("--json", action="store_true")
    p_auto.add_argument("--script", action="store_true")
    p_auto.add_argument("--model-path")
    p_auto.add_argument("--port", type=int, default=8000)
    p_auto.add_argument("--served-model-name")
    p_auto.set_defaults(func=_cmd_auto)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
