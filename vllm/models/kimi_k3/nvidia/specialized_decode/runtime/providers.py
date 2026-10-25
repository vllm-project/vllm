# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer attention providers (input projection, KDA/MLA core, o-projection) and
the helpers ``SpecializedDecode.build_providers`` uses."""

from typing import Any

import torch

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from ..cutedsl.gemm.projection import PreparedProjection
from ..cutedsl.kda.prepared import PreparedKda
from ..cutedsl.mla.dcp import DcpBox
from ..cutedsl.mla.prepared import PreparedMla
from ..cutedsl.oproj.prepared import PreparedOproj

# Decode batches (tokens) served by the prepared providers: every batch up to 8, the
# padded CUDA-graph batches and each further captured size; others run stock.
BATCHES = (*range(1, 9), 16, 32, 64)
PREPARED_MAX = BATCHES[-1]


def projection_policies(tp, rows, batches):
    """Split-K (M tile, N tile, split, stages) per execution rows' next tested bin."""
    policies = {}
    for batch in batches:
        n = next(b for b in (8, 16, 32, 64) if batch <= b)
        if rows == 6288 and (tp == 8 or n > 8):
            tactic = (64, 64, 1, 6) if n == 64 else (128, n, 2, 5 if n == 32 else 6)
        elif tp == 8 and n == 8:
            # TP8 MLA (3648 rows): 128-row tiles give 116 one-per-SM CTAs in one wave;
            # 64-row tiles (228 CTAs) ran two waves on 152 SMs (10.5 vs 8.9 us cold).
            tactic = (128, 8, 4, 6)
        else:  # TP16 KDA B64: a sixth stage, -0.7 us per layer in the KDA chain
            stages = {8: 8, 16: 10, 32: 8, 64: 6 if rows == 3216 else 5}[n]
            tactic = (64, n, 4 if n == 8 else 2, stages)
        policies[batch] = tactic
    return policies


class SkinnyProjection:
    """B1/B2 skinny GEMM; static K loads two weight tiles before the PDL wait."""

    def __init__(self, weight, fallback):
        self.weight, self.fallback, k = weight, fallback, weight.shape[1]
        tactics = {1: (128, 4, 1), 2: (64, 4, 2)}
        self.configs = {m: SkinnyGemmConfig(m, *t, 8, k) for m, t in tactics.items()}
        for m, cfg in self.configs.items():  # compile before capture
            shape_dynamic_skinny_gemm(weight.new_zeros(m, k), weight, cfg, None)

    def run(self, x, out):
        cfg = self.configs.get(x.shape[0])
        if cfg is None:
            return self.fallback.run(x, out)
        return shape_dynamic_skinny_gemm(x, self.weight, cfg, None)


def scratch(runtime, name, shape, dtype=torch.bfloat16):
    if name not in runtime.buffers:
        runtime.buffers[name] = torch.empty(shape, device=runtime.device, dtype=dtype)
    return runtime.buffers[name]


class Attention:
    """Input projection -> KDA/MLA core -> o-projection with its fused AttnRes."""

    def __init__(self, decoder, runtime, prefetch, mailbox, small_batch, twoshot=None):
        attention, tp = decoder.self_attn, runtime.tp
        self.kind = kind = "kda" if hasattr(attention, "in_proj_qkvgfab") else "mla"
        name = "in_proj_qkvgfab" if kind == "kda" else "fused_qkv_a_g_proj"
        weight = getattr(attention, name).weight
        pitch = weight.shape[0]
        policies = projection_policies(tp, pitch, runtime.batches)
        split_k = PreparedProjection(weight, policies).prepare()
        self.projection = SkinnyProjection(weight, split_k)
        self.core: Any  # PreparedKda or PreparedMla
        if kind == "kda":
            # TP8: gate mode 1 at B1, the scalar mode 2 at B2, the MMA mode 0 from B3.
            mode = lambda b: (1 if b == 1 else 2 if b == 2 else 0) if tp == 8 else 0
            policies = {b: (pitch, mode(b)) for b in runtime.batches}
            q = runtime.q  # verify steps: B requests x q tokens
            verify = [b // q for b in runtime.batches if q > 1 and b % q == 0]
            policies.update({(b, q): (pitch, int(tp == 8)) for b in verify})
            self.core = PreparedKda(attention, policies, is_conv_state_dim_first())
        else:
            heads = attention.num_local_heads
            query = scratch(runtime, "mla.query", (64, heads, 576), torch.uint8)
            policy = dict(max_context=runtime.max_context, core=runtime.mla_core)
            policy.update(queries=runtime.queries)
            if runtime.dcp > 1:
                policy.update(
                    dcp=dcp_box(runtime, heads), parity=runtime.mla_layers % 2
                )
            runtime.mla_layers += 1
            pitches = {b: pitch for b in runtime.batches}
            self.core = PreparedMla(attention, pitches, query, **policy)
        parity = decoder.layer_idx % 2
        front: tuple | None = None
        if decoder.is_moe_layer:
            front = small_batch.front_views[parity], small_batch.hint_views[parity]
            front += (small_batch.batches,)
        self.output = PreparedOproj(
            decoder, mailbox, runtime.batches, kind == "mla", front, twoshot
        )
        projected = scratch(runtime, kind + ".projected", (64, pitch))
        local = scratch(runtime, "attention.local", (64, 12288 // tp))
        self.prefetch, self.layer = prefetch, decoder
        self.storage = {b: (projected[:b], local[:b]) for b in range(1, 65)}

    def prepare_layout(self, caches):
        if self.kind == "kda":
            self.core.prepare_caches(caches)
        else:
            self.core.prepare_cache(caches[0])

    def run(self, x, out, prefix, blocks, caches, metadata, slot_mapping, q=1):
        b = x.shape[0]
        projected, local = self.storage[b]
        projected = self.projection.run(x, projected)
        if self.kind == "kda":
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
            if q > 1:  # uniform q-token verify rows
                slots = metadata.spec_state_indices_tensor
                accepted = metadata.num_accepted_tokens
                self.core.run(
                    projected, local, caches=caches, slots=slots, accepted=accepted
                )
            else:
                slots = metadata.non_spec_state_indices_tensor
                self.core.run(projected, local, caches=caches, slots=slots)
        else:
            decode = metadata.decode
            tables, lengths = decode.block_table, decode.seq_lens
            # Request metadata shorter than b: the core runs the prepared batch b
            # with the missing rows as empty requests.
            rows = min(b, q * min(lengths.numel(), tables.shape[0]))
            if rows < b:
                local[rows:].zero_()
            slots = slot_mapping[self.layer.self_attn.layer_name]
            mla = dict(cache=caches[0], block_table=tables, lengths=lengths)
            # DCP: lengths are the local shard's, totals the full sequences'.
            mla.update(totals=decode.dcp_tot_seq_lens)
            self.core.run(projected, local, slots=slots, q=q, **mla)
            self.prefetch.fork("attention_moe", self.layer.layer_idx, b)
        self.output.run(local, prefix, blocks, out)
        self.prefetch.fork("oproj_down", self.layer.layer_idx, b)
        return out


def dcp_box(runtime, heads):
    """The model's DCP mailboxes (one set for every MLA layer)."""
    if runtime.dcp_box is None:
        from vllm.distributed import get_dcp_group

        group = get_dcp_group()
        assert group.world_size == runtime.dcp == runtime.tp
        from vllm.config import get_current_vllm_config

        args = group.device_group, runtime.device, group.world_size, group.rank_in_group
        parallel = get_current_vllm_config().parallel_config
        interleave = parallel.cp_kv_cache_interleave_size
        runtime.dcp_box = DcpBox(*args, heads, interleave=interleave)
    return runtime.dcp_box


def pack_front(moe, rank, tp):
    """[this rank's router rows (padded to 16); this rank's latent down shard]."""
    router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
    rows, shard = 896 // tp, 3584 // tp
    router = router[rank * rows : (rank + 1) * rows]
    down = down[rank * shard : (rank + 1) * shard]
    return torch.cat((router, router.new_zeros((-rows % 16, 7168)), down)).detach()
