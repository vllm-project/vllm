# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which implementation each stage runs for a step of `tokens` tokens (requests x
tokens per request). Every entry is a `batch_table` spec; token counts an entry
leaves out run the stock kernels. `additional_config["policy"]` replaces entries
by name, e.g. `{"moe": "1-4:per_route,8-16:grouped"}`. GB300 runs `DEFAULT`,
Hopper (SM90) runs `HOPPER`, each with its own stage names."""

from collections.abc import Mapping

from .cutedsl.policy import (
    batch_table,
)

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


# The Hopper (SM90) executor's stages (`hopper/`), by token count. Every stage
# kernel holds up to 32 rows (the MoE kernel's row cap): plain decode of 1-32
# requests and MTP5 verify steps of 1-5 requests. "auto" takes the stage
# module's own choice for the token count (e.g. the boundary's
# `DEFAULT_MODES`); a stage-specific name pins one (e.g. the boundary's
# "lamport2"). "" runs a stage stock at every token count; the other stages
# keep their Hopper kernels.
HOPPER = {
    # All-reduce + mHC post + mHC pre + RMSNorm of every backbone boundary.
    "boundary": "1-32:auto",
    # Router, routed and shared experts of the 42 MoE layers.
    "moe": "1-32:auto",
    # The MLP of the 3 dense layers.
    "dense_mlp": "1-32:auto",
    # The 34 KDA layers: in_proj, conv / delta rule / gated norm, o_proj.
    "kda": "1-32:auto",
    # The 11 DSA layers: projections, indexer, KV insert, sparse attention,
    # value up-projection, o_proj.
    "dsa": "1-32:auto",
    # The LM head shard GEMM and the logits all-gather of verify steps.
    "logits_head": "1-32:auto",
    "logits_gather": "1-32:auto",
    # The MTP draft layer: its all-reduce + residual add + RMSNorm boundaries,
    # and the DSA and MoE stages above on its layer.
    "draft_boundary": "1-32:auto",
    "draft_dsa": "1-32:auto",
    "draft_moe": "1-32:auto",
}


class Policy:
    def __init__(
        self,
        overrides: Mapping[str, str] | None = None,
        default: Mapping[str, str] = DEFAULT,
    ):
        overrides = dict(overrides or {})
        unknown = overrides.keys() - default.keys()
        if unknown:
            raise ValueError(f"Unknown specialized decode policy entries: {unknown}")
        specs = dict(default) | overrides
        self.tables = {stage: batch_table(spec) for stage, spec in specs.items()}

    def __call__(self, stage: str, tokens: int) -> str | None:
        """The implementation `stage` runs at `tokens`, None for stock."""
        return self.tables[stage].get(tokens)

    def enabled(self, stage: str) -> bool:
        return bool(self.tables[stage])

    def tokens(self, stage: str, choice: str) -> set[int]:
        return {m for m, c in self.tables[stage].items() if c == choice}
