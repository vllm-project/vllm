# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's DSpark drafter (``decode_backend=latency_specialized``).

What changes against the stock ``K3DSparkForCausalLM`` (all exact):
- ``context_proj`` (5 x 7168 -> 7168, 514 MB BF16) is replicated on every TP rank
  in the stock model, so each TP8 / DCP8 rank streams all of it every draft step.
  Here it is column-parallel with its output all-gathered: each rank streams 1/TP
  of the weight (same dot products, full K per output column) and one
  [R * (k + 1), 7168 / TP] BF16 all-gather rebuilds the rows.
- every drafter linear and the lm_head shard run on ``linears.DraftGemm``.
- the ``k`` sequential Markov steps run as one launch each (``markov.py``).
- the draft block's attention folds its tokens into heads (``attention.py``).
- at TP > 1 (``comm.py``, Lamport mailboxes on multicast symmetric memory): the
  context projection's and the lm_head's all-gathers, and the embedding's
  all-reduce fused with the first RMSNorm; under DCP (= TP) the draft attention's
  query gather and LSE combine + W_UV on the specialized decode's DCP mailboxes
  (``attention.DraftDcp``). ``forward`` replaces ``K3DSparkModel.forward`` for the
  batches these serve (the stock forward otherwise). The draft layers' all-reduce
  + RMSNorm sites stay on flashinfer's fused one-shot (``LAYER_AR_ROWS``).

All of it lives here: ``register`` makes vLLM build ``SpecializedK3DSparkForCausalLM``
(a subclass of the stock drafter) for the DSpark draft architecture, and the stock
``dspark_mla.py`` / ``mla.py`` stay unchanged. The column-parallel context projection
replaces the stock replicated one at construction, before the weights load (the
replicated one is allocated and freed once, at load).
"""

from functools import cache

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel

# The DSpark draft checkpoint's architecture (``ModelRegistry``).
DRAFT_ARCH = "K3DSparkModel"


def register() -> None:
    """Serve the DSpark draft with ``SpecializedK3DSparkForCausalLM`` in this
    process (``ModelRegistry.register_model``, vLLM's model-class extension point).

    ``specialized_decode_config`` calls it when the deployment selects the
    specialized decode: the target model is built (and calls it) before the
    speculator loads the drafter. A draft class resolved earlier in this process
    (cached by ``get_model_architecture``) is dropped so the load resolves again."""
    from vllm.model_executor.model_loader import utils as loader_utils
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(DRAFT_ARCH, SpecializedK3DSparkForCausalLM)
    cached = loader_utils._MODEL_ARCH_BY_HASH
    for key, (cls, _) in list(cached.items()):
        if cls is K3DSparkForCausalLM:
            del cached[key]


class SpecializedK3DSparkModel(K3DSparkModel):
    """``K3DSparkModel`` whose forward runs ``specialized_forward`` (the drafter's
    collectives, set at load by ``install_collectives``) first; its None (a batch
    it does not serve) or no hook runs the stock forward."""

    specialized_forward = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.specialized_forward is not None:
            out = self.specialized_forward(input_ids, positions, inputs_embeds)
            if out is not None:
                return out
        return super().forward(input_ids, positions, inputs_embeds)


class SpecializedK3DSparkForCausalLM(K3DSparkForCausalLM):
    """The stock DSpark drafter with the changes listed above, built in place of
    ``K3DSparkForCausalLM`` (``register``). Same weights, same checkpoint names."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        backbone = self.model
        backbone.__class__ = SpecializedK3DSparkModel
        proj = context_proj(backbone, maybe_prefix(prefix, "model"))
        if proj is not None:
            backbone.context_proj = proj
        for layer in backbone.layers:
            install_draft_fold(layer.self_attn)
        # What the drafter's load-time reservations are sized by (after_loading).
        self.draft_shape = draft_shape(vllm_config)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from .linears import draft_logits

        return draft_logits(self, hidden_states)

    def sample_sequential(self, speculator, num_reqs, head_hidden) -> bool:
        """The speculator asks the model for its own Markov sampling first."""
        from .markov import sample_sequential as fused

        return fused(speculator, self, num_reqs, head_hidden)

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        after_loading(self)


def context_proj(backbone, prefix: str):
    """The draft's context projection column-parallel with a gathered output when
    TP > 1 divides it, replacing the stock replicated one before the weights load
    (the checkpoint's ``context_proj.weight`` is sharded by its loader); None keeps
    the stock one."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    config = backbone.config
    input_size = config.target_hidden_size * config.num_target_layers
    output_size = config.hidden_size
    tp = get_tensor_model_parallel_world_size()
    if tp <= 1 or output_size % tp != 0:
        return None
    return _gathered_linear()(
        input_size,
        output_size,
        bias=False,
        gather_output=False,
        return_bias=False,
        quant_config=backbone.quant_config,
        prefix=maybe_prefix(prefix, "context_proj"),
    )


def install_draft_fold(attn) -> None:
    """Fold the draft block's tokens into heads in the layer's MLA decode
    (``attention.draft_forward_mqa``: one decode row per request instead of one per
    token; the stock ``forward_mqa`` for a batch it does not fold)."""
    from .attention import draft_forward_mqa

    if not attn.non_causal_multi_token_decode:
        return
    attn.draft_fold = draft_forward_mqa  # read by DraftDcp.eligible
    attn.impl.__class__ = _folded_impl(type(attn.impl))


@cache
def _folded_impl(base):
    """``base`` (the layer's MLA impl class) whose ``forward_mqa`` runs the folded
    draft decode first. Keeps the class name (logs, backend checks)."""
    from .attention import draft_forward_mqa

    class Folded(base):
        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
            out = draft_forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
            if out is None:
                out = super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
            return out

    Folded.__name__ = base.__name__
    Folded.__qualname__ = base.__qualname__
    return Folded


@cache
def _gathered_linear():
    """``ColumnParallelLinear(gather_output=True)`` whose all-gather is the drafter's
    (``comm.DrafterComm.gather_context``, attached at load), NCCL's otherwise."""
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    class GatheredColumnParallelLinear(ColumnParallelLinear):
        comm = None

        def forward(self, input_):
            shard = super().forward(input_)
            out = None if self.comm is None else self.comm.gather_context(shard)
            return tensor_model_parallel_all_gather(shard) if out is None else out

    return GatheredColumnParallelLinear


def draft_shape(vllm_config: VllmConfig) -> dict | None:
    """What the drafter's load-time reservations are sized by: the largest draft
    batch (requests), the block length ``k``, the local attention heads and the
    plan length of captured graphs; None when the draft attention cannot run CAKE
    (DCP returns an LSE: the cute-dsl kernel)."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    spec = vllm_config.speculative_config
    if spec is None or vllm_config.parallel_config.decode_context_parallel_size > 1:
        return None
    if not str(spec.kv_cache_dtype or "").startswith("fp8"):
        return None
    heads = spec.draft_model_config.hf_config.num_attention_heads
    return dict(
        max_reqs=vllm_config.scheduler_config.max_num_seqs,
        k=spec.num_speculative_tokens,
        heads=heads // get_tensor_model_parallel_world_size(),
        max_seq_len=vllm_config.model_config.max_model_len,
    )


def after_loading(model) -> None:
    """Route the drafter's linears through DraftGemm and reserve the folded draft
    attention's CAKE workspace for the largest draft batch (``model`` is a
    specialized ``K3DSparkForCausalLM``)."""
    from vllm.logger import init_logger

    from . import linears
    from .attention import reserve_cake_workspace

    n = linears.install(model.model)
    shape = getattr(model, "draft_shape", None)
    reserved = 0
    if shape is not None:
        device = next(model.parameters()).device
        reserved = reserve_cake_workspace(device, **shape)
    comm, dcp = install_collectives(model)
    init_logger(__name__).info_once(
        "Specialized DSpark drafter: %d linears on DraftGemm, CAKE workspace %.0f MiB, "
        "collectives %s, DCP attention %s",
        n,
        reserved / 2**20,
        "mailbox" if comm is not None else "stock",
        "mailbox" if dcp is not None else "stock",
    )


def install_collectives(model):
    """Build the drafter's mailboxes (collective: every TP rank runs it at load, in
    the same order) and route the forward, the context projection and the lm_head
    gather through them. Returns (DrafterComm or None, DraftDcp or None)."""
    from vllm.distributed import get_tp_group

    from . import comm as comm_module

    tp = get_tp_group()
    comm = comm_module.build(model, tp)
    if comm is None:
        return None, None
    backbone = model.model
    context = backbone.context_proj
    if hasattr(type(context), "comm"):
        context.comm = comm
    model._drafter_comm = comm
    dcp = _build_dcp(backbone)
    model._drafter_dcp = dcp
    embed: list = []  # the target embedding is aliased after this load: lazily

    def specialized_forward(ids, positions, embeds):
        if not embed:
            embed.append(_prepared_embedding(backbone, tp))
        return forward(backbone, comm, dcp, embed[0], ids, positions, embeds)

    backbone.specialized_forward = specialized_forward
    return comm, dcp


def _build_dcp(backbone):
    """The draft attention's DCP path when DCP = TP (rank d owns head block d)."""
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import get_dcp_group
    from vllm.logger import init_logger

    from .attention import DraftDcp
    from .comm import ROWS_CAP

    attentions = [layer.self_attn for layer in backbone.layers]
    size = attentions[0].dcp_world_size
    tp = get_tp_group()
    if size <= 1:
        return None
    group = get_dcp_group()
    if group.world_size != tp.world_size or group.rank_in_group != tp.rank_in_group:
        return None
    heads = attentions[0].num_local_heads
    if not all(DraftDcp.eligible(a, heads) for a in attentions):
        return None
    device = next(backbone.parameters()).device
    try:
        dcp = DraftDcp(
            group.device_group,
            device,
            size,
            group.rank_in_group,
            attentions[0].num_local_heads,
            ROWS_CAP,
        )
    except Exception as exc:  # capability fallback: the stock DCP collectives
        init_logger(__name__).warning_once("DSpark draft DCP stays stock: %s", exc)
        return None
    dcp.prepare(attentions)
    return dcp


def _prepared_embedding(backbone, tp):
    """The vocab-parallel embedding's local lookup (``cutedsl.common.embedding``,
    zero rows for ids another rank owns) whose all-reduce ``forward`` fuses with
    the first draft layer's RMSNorm; None keeps the stock embedding."""
    from ..cutedsl.common.embedding import PreparedEmbedding

    embedding = backbone.embed_tokens
    try:
        return PreparedEmbedding(embedding, tp.world_size, tp.rank_in_group)
    except (ValueError, AttributeError) as exc:
        from vllm.logger import init_logger

        init_logger(__name__).info_once("DSpark draft embedding stays stock: %s", exc)
        return None


# Rows up to which the embedding's all-reduce + first RMSNorm run fused (ArNorm,
# ~8.4 us at 16 rows) instead of the stock lookup + all-reduce + RMSNorm (~14 us);
# above it the fused one-shot trails (13.7 us at 48 rows).
EMBED_ROWS = 16
# Rows up to which the draft layers' all-reduce + RMSNorm sites run ArNorm instead
# of flashinfer's fused one-shot. Per call in isolation (8 GB300 ranks, us, ArNorm /
# flashinfer): 6.7 / 6.6 at 4 rows, 8.4 / 7.2 at 16, 13.7 /
# 8.2 at 48.
LAYER_AR_ROWS = 0


def forward(backbone, comm, dcp, embed, input_ids, positions, inputs_embeds):
    """``K3DSparkModel.forward`` with the embedding's all-reduce fused into the
    first RMSNorm (ArNorm) and the draft attention on ``dcp``, or None (the stock
    forward) for a batch these do not serve. Same layers, same order."""
    from vllm.models.common.ops.fused_allreduce_rms_norm import (
        fused_allreduce_rms_norm,
    )

    from .comm import ROWS_CAP

    rows = input_ids.shape[0] if inputs_embeds is None else inputs_embeds.shape[0]
    if not 1 <= rows <= ROWS_CAP:
        return None
    layers = backbone.layers

    site = [1]  # the embedding's ArNorm takes parity 0

    def reduce(x, residual, norm):
        if rows <= LAYER_AR_ROWS:
            out = comm.ar_norm(x, residual, norm, site[0] % 2)
            site[0] += 1
            if out is not None:
                return out
        return fused_allreduce_rms_norm(x, residual, norm)

    first_norm = layers[0].input_layernorm
    if (
        inputs_embeds is None
        and embed is not None
        and rows <= EMBED_ROWS
        and first_norm.weight.dtype == torch.bfloat16
    ):
        partial = torch.empty(
            (rows, embed.weight.shape[1]),
            dtype=embed.weight.dtype,
            device=input_ids.device,
        )
        embed.kernels[input_ids.dtype](
            input_ids, embed.weight, partial, embed.start, embed.vocab
        )
        out = comm.ar_norm(partial, None, first_norm, 0)
        assert out is not None, (tuple(partial.shape), partial.dtype)  # BF16 checked
        hidden, residual = out
    else:
        if inputs_embeds is None:
            inputs_embeds = backbone.embed_input_ids(input_ids)
        residual = inputs_embeds
        hidden = first_norm(inputs_embeds)
    for index, layer in enumerate(layers):
        if index > 0:
            hidden, residual = reduce(hidden, residual, layer.input_layernorm)
        out = None
        if dcp is not None:
            out = dcp.forward(layer.self_attn, positions, hidden, index % 2)
        if out is None:
            out = layer.self_attn(positions=positions, hidden_states=hidden)
        hidden, residual = reduce(out, residual, layer.post_attention_layernorm)
        hidden = layer.mlp(hidden)
    hidden, _ = reduce(hidden, residual, backbone.final_norm)
    return hidden
