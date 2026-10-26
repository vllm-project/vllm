# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-layer shapes of current flagship models, for backend test matrices.

Values are copied from each model's Hugging Face ``config.json`` (read on
2026-10-24). Only models whose layers run through a dense MHA/GQA backend are
listed; MLA and sparse-attention models (DeepSeek V3/V3.2/V4/V4.1, GLM-5.x,
Kimi K3, MiniMax M3's sparse layers) belong to the MLA/sparse backend suites.
Re-read the configs rather than hand-editing a number.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttnShape:
    """One attention layer kind of one model, at a given TP degree."""

    model: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    sliding_window: int | None = None
    tp: int = 1
    sinks: bool = False

    @property
    def id(self) -> str:
        name = self.model.split("/")[-1]
        window = f"-swa{self.sliding_window}" if self.sliding_window else ""
        sinks = "-sinks" if self.sinks else ""
        return (
            f"{name}-tp{self.tp}-h{self.num_heads}kv{self.num_kv_heads}"
            f"d{self.head_size}{window}{sinks}"
        )

    def shard(self, tp: int) -> "AttnShape":
        """Per-rank shape; KV heads are replicated once tp exceeds them."""
        assert self.num_heads % tp == 0
        return AttnShape(
            self.model,
            self.num_heads // tp,
            max(self.num_kv_heads // tp, 1),
            self.head_size,
            self.sliding_window,
            tp,
            self.sinks,
        )


QWEN3_235B = AttnShape("Qwen/Qwen3-235B-A22B", 64, 4, 128)
QWEN3_8_27B = AttnShape("Qwen/Qwen3.8-27B", 24, 4, 256)
QWEN3_8_2_4T = AttnShape("Qwen/Qwen3.8-2.4T-A95B", 64, 4, 256)
# Served by its own QSA backend; kept as a GQA-12 head-256 shape.
QWEN3_8_FLASH_NEXT = AttnShape("Qwen/Qwen3.8-Flash-Next", 24, 2, 256)
GEMMA4_31B_SLIDING = AttnShape("google/gemma-4-31B-it", 32, 16, 256, 1024)
GEMMA4_31B_GLOBAL = AttnShape("google/gemma-4-31B-it", 32, 4, 512)
GEMMA4_26B_SLIDING = AttnShape("google/gemma-4-26B-A4B-it", 16, 8, 256, 1024)
GEMMA4_26B_GLOBAL = AttnShape("google/gemma-4-26B-A4B-it", 16, 2, 512)
# Gated repo (needs HF_TOKEN); values checked against its config.json.
MUSE_SPARK_SLIDING = AttnShape(
    "meta-models/Muse-Spark-Open-816B-A42B", 128, 16, 64, 2048
)
MUSE_SPARK_FULL = AttnShape("meta-models/Muse-Spark-Open-816B-A42B", 128, 16, 64)
# Layers 0-2 of MiniMax M3 are dense GQA (sparse_attention_freq == 0); the rest
# are block-sparse. MiniMax M3.1-Flash has no dense layers.
MINIMAX_M3 = AttnShape("MiniMaxAI/MiniMax-M3", 64, 4, 128)
# gpt-oss-20b has the same attention config (64/8/64, window 128, sinks).
GPT_OSS_SLIDING = AttnShape("openai/gpt-oss-120b", 64, 8, 64, 128, sinks=True)
GPT_OSS_FULL = AttnShape("openai/gpt-oss-120b", 64, 8, 64, sinks=True)

ALL_SHAPES = [
    QWEN3_235B,
    QWEN3_8_27B,
    QWEN3_8_2_4T,
    QWEN3_8_FLASH_NEXT,
    GEMMA4_31B_SLIDING,
    GEMMA4_31B_GLOBAL,
    GEMMA4_26B_SLIDING,
    GEMMA4_26B_GLOBAL,
    MUSE_SPARK_SLIDING,
    MUSE_SPARK_FULL,
    MINIMAX_M3,
    GPT_OSS_SLIDING,
    GPT_OSS_FULL,
]

# Per-rank shapes that change the H/KV ratio or replicate KV heads.
TP_SHARDED_SHAPES = [
    QWEN3_235B.shard(8),
    QWEN3_8_27B.shard(4),
    MUSE_SPARK_FULL.shard(8),
    MUSE_SPARK_SLIDING.shard(8),
    GEMMA4_31B_SLIDING.shard(2),
]


def shapes_supported_by(head_sizes: list[int]) -> list[AttnShape]:
    """Shapes a backend with the given head sizes must run (TP=1 and sharded)."""
    return [s for s in ALL_SHAPES + TP_SHARDED_SHAPES if s.head_size in head_sizes]


def shapes_rejected_by(head_sizes: list[int]) -> list[AttnShape]:
    return [s for s in ALL_SHAPES if s.head_size not in head_sizes]


@dataclass(frozen=True)
class ParallelLayout:
    """One deployment of a model on ``gpus`` GPUs (= tp * pp * pcp * dp).

    ``ep`` is the expert-parallel group size (tp * pcp * dp) or 1 when EP is
    off; DCP reuses TP ranks and shards the KV cache, so each rank holds about
    1/dcp of the context. PCP alone replicates the KV cache. See
    attn-test-work/parallel-layouts.md for the vLLM rules behind each reason.
    """

    model: str
    gpus: int
    tp: int
    dp: int = 1
    ep: int = 1
    pp: int = 1
    pcp: int = 1
    dcp: int = 1
    valid: bool = True
    reason: str = ""

    def __post_init__(self):
        assert self.gpus == self.tp * self.pp * self.pcp * self.dp, self
        assert self.ep in (1, self.tp * self.pcp * self.dp), self
        assert self.valid == (not self.reason), self


def _no(model: str, gpus: int, tp: int, reason: str, **kw) -> ParallelLayout:
    return ParallelLayout(model, gpus, tp, valid=False, reason=reason, **kw)


_L = ParallelLayout
Q235, Q27, Q24T = (s.model for s in (QWEN3_235B, QWEN3_8_27B, QWEN3_8_2_4T))
QFN = QWEN3_8_FLASH_NEXT.model
G31, G26 = GEMMA4_31B_GLOBAL.model, GEMMA4_26B_GLOBAL.model
MUSE, MM3, GPT_OSS = MUSE_SPARK_FULL.model, MINIMAX_M3.model, GPT_OSS_FULL.model
MM31F = "MiniMaxAI/MiniMax-M3.1-Flash"
DSV3, DSV32 = "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.2"
DSV4P, DSV4F = "deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"
DSV41F = "deepseek-ai/DeepSeek-V4.1-Flash"
GLM52, GLM53, GLM53F = "zai-org/GLM-5.2", "zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"
KIMI_K3 = "moonshotai/Kimi-K3"

KV_LE_TP = "GQA DCP needs tp > total KV heads (ModelConfig)"
DENSE_NO_CP = "no ROCm dense backend supports DCP/PCP"
NOT_MOE = "expert parallel needs an MoE model"
QSA_NO_CP = "Qwen4Exp QSA rejects DCP/PCP"
SPARSE_NO_DCP = "ROCm sparse MLA (+indexer) does not support DCP"
SWA_NO_CP = "DCP/PCP reject SWA/compressed KV groups"
DSV41_NO_PCP = "DeepSeek-V4.1 PCP is CUDA-only"
HYBRID_NO_PCP = "PCP rejects linear-attention (Mamba/KDA) KV groups"
MLA_NO_PCP_DCP = "generic MLA layer does not support PCP+DCP"
MUSE_DP1 = "MuseSpark1x MoE on ROCm requires DP1"

PARALLEL_LAYOUTS = [
    _L(Q235, 4, 4, ep=4),
    _L(Q235, 4, 2, dp=2, ep=4),
    _L(Q235, 4, 2, pp=2),
    _no(Q235, 4, 4, KV_LE_TP, dcp=2),
    _L(Q235, 8, 8, ep=8),
    _L(Q235, 8, 4, dp=2, ep=8),
    _L(Q235, 8, 1, dp=8, ep=8),
    _L(Q235, 8, 4, pp=2),
    _no(Q235, 8, 8, DENSE_NO_CP, dcp=2),
    _no(Q235, 8, 4, DENSE_NO_CP, pcp=2),
    _L(Q27, 1, 1),
    _L(Q27, 2, 2),
    _L(Q27, 4, 4),
    _no(Q27, 4, 4, NOT_MOE, ep=4),
    _L(Q27, 8, 8),
    _L(Q27, 8, 4, dp=2),
    _no(Q27, 8, 8, DENSE_NO_CP, dcp=2),
    _L(Q24T, 8, 8, ep=8),
    _L(Q24T, 8, 4, dp=2, ep=8),
    _L(Q24T, 8, 4, pp=2),
    _no(Q24T, 8, 8, DENSE_NO_CP, dcp=2),
    _L(QFN, 1, 1),
    _L(QFN, 2, 2),
    _L(QFN, 4, 4),
    _L(QFN, 8, 8, ep=8),
    _L(QFN, 8, 4, dp=2, ep=8),
    _no(QFN, 4, 4, QSA_NO_CP, dcp=2),
    _L(DSV41F, 4, 4, ep=4),
    _L(DSV41F, 8, 8, ep=8),
    _L(DSV41F, 8, 4, dp=2, ep=8),
    _no(DSV41F, 8, 8, SWA_NO_CP, dcp=2),
    _no(DSV41F, 8, 4, DSV41_NO_PCP, pcp=2),
    _L(DSV4P, 4, 4, ep=4),
    _L(DSV4P, 8, 8, ep=8),
    _L(DSV4P, 8, 4, dp=2, ep=8),
    _no(DSV4P, 8, 8, SWA_NO_CP, dcp=2),
    _L(DSV4F, 4, 4, ep=4),
    _L(DSV4F, 8, 8, ep=8),
    _L(MM3, 2, 2),
    _L(MM3, 4, 4, ep=4),
    _L(MM3, 8, 8, ep=8),
    _L(MM3, 8, 4, dp=2, ep=8),
    _no(MM3, 8, 8, DENSE_NO_CP, dcp=2),
    _no(MM3, 8, 4, DENSE_NO_CP, pcp=2),
    _L(MM31F, 2, 2),
    _L(MM31F, 4, 4, ep=4),
    _L(MM31F, 8, 8, ep=8),
    _L(GLM52, 8, 8, ep=8),
    _L(GLM52, 8, 4, dp=2, ep=8),
    _no(GLM52, 8, 8, SPARSE_NO_DCP, dcp=2),
    _L(GLM53, 4, 4, ep=4),
    _L(GLM53, 8, 8, ep=8),
    _L(GLM53, 8, 4, dp=2, ep=8),
    _L(GLM53F, 4, 4, ep=4),
    _L(GLM53F, 8, 8, ep=8),
    _L(GLM53F, 8, 4, dp=2, ep=8),
    _no(GLM53F, 8, 8, SPARSE_NO_DCP, dcp=2),
    _no(GLM53F, 8, 4, HYBRID_NO_PCP, pcp=2),
    _L(G31, 1, 1),
    _L(G31, 2, 2),
    _no(G31, 2, 2, KV_LE_TP, dcp=2),
    _L(G26, 1, 1),
    _L(G26, 2, 2, ep=2),
    _L(G26, 2, 1, dp=2, ep=2),
    _L(MUSE, 8, 8, ep=8),
    _L(MUSE, 8, 4, pp=2),
    _no(MUSE, 8, 4, MUSE_DP1, dp=2, ep=8),
    _no(MUSE, 8, 8, KV_LE_TP, dcp=2),
    _L(DSV3, 8, 8, ep=8),
    _L(DSV3, 8, 4, dp=2, ep=8),
    _L(DSV3, 8, 1, dp=8, ep=8),
    _L(DSV3, 8, 8, ep=8, dcp=8),
    _L(DSV3, 8, 4, pcp=2),
    _no(DSV3, 8, 4, MLA_NO_PCP_DCP, pcp=2, dcp=2),
    _L(DSV32, 8, 8, ep=8),
    _L(DSV32, 8, 4, dp=2, ep=8),
    _no(DSV32, 8, 8, SPARSE_NO_DCP, dcp=2),
    _L(KIMI_K3, 4, 4, ep=4),
    _L(KIMI_K3, 8, 8, ep=8),
    _L(KIMI_K3, 8, 8, ep=8, dcp=8),
    _no(KIMI_K3, 8, 4, HYBRID_NO_PCP, pcp=2),
    _L(GPT_OSS, 1, 1),
]

# Layers that run through the generic dense backends (ROCM_ATTN, ROCM_AITER_FA,
# TRITON_ATTN); Qwen3.8-Flash-Next's QSA layers do not.
DENSE_SHAPES = [s for s in ALL_SHAPES if s is not QWEN3_8_FLASH_NEXT]


def dense_rank_shapes(
    layouts: list[ParallelLayout] | None = None,
) -> list[AttnShape]:
    """Distinct per-rank dense-MHA shapes implied by the valid layouts.

    Shapes are deduplicated on what a kernel sees (heads, KV heads, head size,
    window, sinks); the first model/TP that produced each one is kept.
    """
    shapes: dict[tuple, AttnShape] = {}
    for layout in PARALLEL_LAYOUTS if layouts is None else layouts:
        if not layout.valid:
            continue
        for base in DENSE_SHAPES:
            if base.model == layout.model:
                s = base.shard(layout.tp)
                key = (s.num_heads, s.num_kv_heads, s.head_size)
                shapes.setdefault(key + (s.sliding_window, s.sinks), s)
    return list(shapes.values())


def cp_rejected_layouts(models: set[str] | None = None) -> list[ParallelLayout]:
    """Rejected layouts that use DCP or PCP, optionally for given models."""
    return [
        layout
        for layout in PARALLEL_LAYOUTS
        if not layout.valid
        and layout.dcp * layout.pcp > 1
        and (models is None or layout.model in models)
    ]
