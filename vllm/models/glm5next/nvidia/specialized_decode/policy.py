# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which implementation each stage runs for a step of `tokens` tokens (requests x
tokens per request). Every entry is a `batch_table` spec; token counts an entry
leaves out run the stock kernels. `additional_config["policy"]` replaces entries
by name, e.g. `{"moe": "1-4:per_route,8-16:grouped"}`."""

from collections.abc import Mapping

from .cutedsl.policy import batch_table

DEFAULT = {
    # Fused all-reduce + mHC boundary (all-reduce + norm in the draft), by
    # tokens per 8-CTA cluster. GB300 holds 15 such clusters at once; each
    # extra token per cluster costs ~1.2 us (standalone 7.6 / 8.9 / 10.0 / 11.1
    # / 12.6 us at 8 / 16 / 32 / 48 / 64 tokens; stock all-reduce + mHC ~25 us
    # at 64).
    "boundary": "1-15:1,16-30:2,31-45:3,46-60:4,61-64:5",
    # Routed experts: one 16-CTA cluster per route (two CTAs per SM) up to 4
    # tokens, where TRTLLM-Gen overtakes it (42 vs 49 us at 4 tokens, 73 vs 66
    # at 8). From 8 tokens the persistent expert-grouped kernel ("grouped",
    # with the shared expert) wins or ties: MoE layer 74 vs 76 us at 8 tokens,
    # 94 vs 98 at 16, 126 vs 140 at 32, 171 vs 170 at 64 (164 vs 179 with MTP3).
    "moe": "1-4:per_route,8-64:grouped",
    # Router logits at 32-64 tokens as the ll_bf16 split-K GEMM (stock: cuBLAS
    # split-K + reduce, ~2 us longer on the critical path).
    "router": "32-64:split_k",
    # Shared expert GEMVs beside the per-route clusters. At four tokens they slow
    # the clusters more than DeepGEMM does (B4 step 4.56 vs 4.37 ms).
    "shared_expert": "1-2:cute",
    # Dense layers' MLP: 7.5 / 8.4 / 11.0 us per layer in situ at 1 / 4 / 8
    # tokens, stock ~20.
    "dense_mlp": "1-2:cute,4:cute,8:cute",
    # KDA core: Kimi K3's fused KdaSplit (conv, F_B, decay, delta rule, gated
    # norm) in place of stock's conv / F_B GEMM / recurrence / norm kernels;
    # "split_gate" also folds in g_b_proj: KDA layer -1.3 / -0.7 us in situ at
    # 32 / 64 tokens (-1.9 / -1.1 with MTP3).
    "kda": "1-16:split,32-64:split_gate",
    # KDA output gate g_b_proj; cuBLAS wins at 8 (1.7 vs 2.0 us with the
    # weights out of L2).
    "kda_gate": "1-4:cute",
    # DSA W_UV value up-projection; cuBLAS's bmm wins at 8 (2.5 vs 3.0 us).
    "dsa_value_up": "1-4:cute",
    # DSA BF16 projections (q/kv a, q_b, o_proj, indexer wq_b) as Kimi K3's
    # split-K GEMM, where gemm.SPLIT_K has a tactic for the shape (cuBLAS
    # elsewhere); 1-2 tokens take the skinny GEMM.
    "dsa_proj": "4-64:split_k",
    # The LM head's logits all-gather as one NVLink kernel straight into the
    # [rows, vocab] layout: in situ 31 / 42 us at 32 / 64 rows, NCCL all-gather
    # + permute copy 63 / 79.
    "logits_gather": "1-64:nvlink",
    # The LM head shard GEMM: the skinny GEMM at 1-2 rows, the split-K GEMM from
    # 4 rows (the MTP head's tactics; in situ 55 -> 51.6 us at 64 rows).
    "logits_head": "1-64:cute",
    # Stages without a per-token choice: on at every planned step, "" runs
    # stock. kda_proj is the KDA skinny in_proj / o_proj GEMM, which picks its
    # own tactic per token count.
    "kda_proj": "1-64:cute",
    "dsa_indexer": "1-64:cute",
    "dsa_attn": "1-64:cute",
    "dsa_kpool": "1-64:cute",
}

_SM90_OVERRIDES = {
    "moe": "1-64:triton",
    "dense_mlp": "1-8:cute",
    "kda": "1-64:split",
    # At four or more rows, use cuBLAS for the vocabulary shard.
    "logits_head": "1-2:cute",
}


def _kda_restrictions(vllm_config) -> list[str]:
    import torch

    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cache = vllm_config.cache_config
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
    )
    reasons = []
    if dtypes != (torch.bfloat16, torch.float32):
        reasons.append(
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent "
            "state (mamba cache dtype)"
        )
    if cache.use_replayssm or cache.use_kda_recoverssm:
        reasons.append("ReplaySSM / RecoverSSM (--use-replayssm)")
    return reasons


def unsupported_platform(vllm_config, overrides: dict[str, str]) -> list[str]:
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_tilelang
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if current_platform.is_cuda():
        if current_platform.is_device_capability_family(90):
            reasons = []
            if any(
                c != "triton" for c in batch_table(overrides.get("moe", "")).values()
            ):
                reasons.append("Hopper MoE choices other than triton")
            if not has_tilelang():
                reasons.append("missing TileLang on Hopper")
            return reasons
        if current_platform.is_device_capability_family(100):
            cache = vllm_config.cache_config
            attention = vllm_config.attention_config
            backend = attention.backend_per_kind.get("mla_attention", attention.backend)
            checks = (
                (
                    cache.cache_dtype not in ("fp8", "fp8_e4m3"),
                    "KV cache dtypes other than FP8 E4M3",
                ),
                (bool(cache.kv_cache_dtype_skip_layers), "mixed KV cache dtypes"),
                (
                    backend not in (None, AttentionBackendEnum.FLASHINFER_MLA_SPARSE),
                    "attention backends other than FLASHINFER_MLA_SPARSE",
                ),
                (attention.hisparse_config is not None, "HiSparse"),
            )
            return [reason for failed, reason in checks if failed] + _kda_restrictions(
                vllm_config
            )
    return ["this GPU (requires NVIDIA SM90 or SM100/103)"]


def _supports_sm90_dsa(vllm_config) -> bool:
    from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention = vllm_config.attention_config
    cache = vllm_config.cache_config
    backend = attention.backend_per_kind.get("mla_attention", attention.backend)
    return (
        backend in (None, AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90)
        and attention.hisparse_config is None
        and FlashInferMLASparseSM90Backend.supports_kv_cache_dtype(cache.cache_dtype)
        and (
            not cache.user_specified_block_size
            or FlashInferMLASparseSM90Backend.supports_block_size(cache.block_size)
        )
        and has_flashinfer_sm90_nope_mla()
    )


def platform_policy(
    vllm_config, overrides: dict[str, str], *, strict: bool
) -> dict[str, str]:
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability_family(90):
        return overrides
    policy = _SM90_OVERRIDES | overrides
    if not _supports_sm90_dsa(vllm_config):
        if strict and policy.get("dsa_attn", True):
            raise ValueError(
                "GLM-5.3-Flash specialized Hopper DSA needs FlashInfer SM90 MLA "
                "with compatible KV cache and HiSparse disabled"
            )
        policy.update((stage, "") for stage in DEFAULT if stage.startswith("dsa_"))
    if policy["kda"] and _kda_restrictions(vllm_config):
        policy["kda"] = ""
    return policy


class Policy:
    def __init__(self, overrides: Mapping[str, str] | None = None):
        overrides = dict(overrides or {})
        unknown = overrides.keys() - DEFAULT.keys()
        if unknown:
            raise ValueError(f"Unknown specialized decode policy entries: {unknown}")
        self.tables = {
            stage: batch_table(spec) for stage, spec in (DEFAULT | overrides).items()
        }

    def __call__(self, stage: str, tokens: int) -> str | None:
        """The implementation `stage` runs at `tokens`, None for stock."""
        return self.tables[stage].get(tokens)

    def enabled(self, stage: str) -> bool:
        return bool(self.tables[stage])

    def tokens(self, stage: str, choice: str) -> set[int]:
        return {m for m, c in self.tables[stage].items() if c == choice}
