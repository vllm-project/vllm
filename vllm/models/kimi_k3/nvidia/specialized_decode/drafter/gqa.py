# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's GQA DSpark drafter (`Qwen3DSparkModel` drafts of
Kimi K3.1, e.g. Inferact/Kimi-K3.1-Pro-DSpark: GQA sliding-window layers).

What changes against the stock `Qwen3DSparkForCausalLM` (all exact up to the
summation order of a GEMM):
- `fc` (the context projection, replicated in the stock model) is
  column-parallel with its output gathered on the drafter's mailbox.
- every drafter linear, the fused context K/V projection and the lm_head shard
  run on `linears.DraftGemm`.
- the `k` sequential Markov steps run fused (`markov.py`).
- the row-parallel `o_proj` / `down_proj` partials are reduced in the RMSNorm
  that follows (`fused_allreduce_rms_norm`, as the compiled stock model fuses
  them), and at small batches the embedding's all-reduce is fused with the first
  RMSNorm (`comm.DrafterComm.ar_norm`).

The draft attention stays the stock sliding-window backend.
"""

import torch

from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm

from .linears import DraftGemm
from .model import EMBED_ROWS, _gathered_linear, _prepared_embedding

DRAFT_ARCH = "Qwen3DSparkModel"


@ignore_torch_compile
class SpecializedQwen3DSparkModel(Qwen3DSparkModel):
    """`Qwen3DSparkModel` whose row-parallel outputs are reduced by the RMSNorm
    after them (`forward`)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        for layer in self.layers:
            layer.self_attn.o_proj.reduce_results = False
            layer.mlp.down_proj.reduce_results = False
        self._context_kv_gemm: DraftGemm | None = None
        self._comm = None
        self._tp = None
        self._embed: list = []  # the target embedding, aliased after the load

    def _project_context_kv_token_major(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        gemm = self._context_kv_gemm
        if (
            gemm is None
            or gemm.weight is not self._fused_kv_weight
            or self._fused_kv_bias is not None
        ):
            return super()._project_context_kv_token_major(
                context_states, num_ctx, num_layers, num_kv_heads, head_dim
            )
        normed = self._normalize_context_states(context_states).contiguous()
        out = torch.empty(num_ctx, gemm.n, dtype=normed.dtype, device=normed.device)
        gemm.run(normed, out)
        return out.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`DFlashQwen3Model.forward` with explicit reductions (see the class)."""
        from .comm import ROWS_CAP

        comm = self._comm
        rows = input_ids.shape[0] if input_embeds is None else input_embeds.shape[0]
        first_norm = self.layers[0].input_layernorm
        embed = None
        if (
            comm is not None
            and input_embeds is None
            and not self.has_separate_mask_embedding
            and 1 <= rows <= min(EMBED_ROWS, ROWS_CAP)
            and first_norm.weight.dtype == torch.bfloat16
        ):
            if not self._embed:
                self._embed.append(_prepared_embedding(self, self._tp))
            embed = self._embed[0]
        if embed is not None:
            partial = torch.empty(
                (rows, embed.weight.shape[1]),
                dtype=embed.weight.dtype,
                device=input_ids.device,
            )
            embed.kernels[input_ids.dtype](
                input_ids, embed.weight, partial, embed.start, embed.vocab
            )
            out = comm.ar_norm(partial, None, first_norm, 0)
            assert out is not None, (tuple(partial.shape), partial.dtype)
            hidden, residual = out
        else:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            residual = input_embeds
            hidden = first_norm(input_embeds)
        for index, layer in enumerate(self.layers):
            if index > 0:
                hidden, residual = fused_allreduce_rms_norm(
                    hidden, residual, layer.input_layernorm
                )
            hidden = layer.self_attn(positions=positions, hidden_states=hidden)
            hidden, residual = fused_allreduce_rms_norm(
                hidden, residual, layer.post_attention_layernorm
            )
            hidden = layer.mlp(hidden)
        hidden, _ = fused_allreduce_rms_norm(hidden, residual, self.norm)
        return hidden


class SpecializedQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """The stock GQA DSpark drafter with the changes listed above, built in place
    of `Qwen3DSparkForCausalLM` (`model.register`). Same weights and names."""

    model_cls = SpecializedQwen3DSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        backbone = self.model
        if getattr(backbone, "use_aux_hidden_state", False):
            proj = _gathered_fc(backbone.fc, maybe_prefix(prefix, "model"))
            if proj is not None:
                backbone.fc = proj

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from .linears import draft_logits

        return draft_logits(self, hidden_states)

    def sample_sequential(self, speculator, num_reqs, head_hidden) -> bool:
        """The speculator asks the model for its own Markov sampling first."""
        from .markov import sample_sequential as fused

        return fused(speculator, self, num_reqs, head_hidden)

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        _after_loading(self)


def _gathered_fc(fc, prefix: str):
    """`fc` column-parallel with a gathered output when TP > 1 divides it,
    replacing the replicated one before the weights load; None keeps it."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    if tp <= 1 or fc.output_size % tp != 0:
        return None
    return _gathered_linear()(
        fc.input_size,
        fc.output_size,
        bias=False,
        gather_output=False,
        return_bias=False,
        params_dtype=fc.params_dtype,
        quant_config=fc.quant_config,
        prefix=maybe_prefix(prefix, "fc"),
    )


def _after_loading(model) -> None:
    """Route the drafter's linears and its fused context K/V projection through
    DraftGemm and build its collectives (every TP rank, in the same order)."""
    from vllm.distributed import get_tp_group
    from vllm.logger import init_logger

    from . import comm as comm_module
    from . import linears

    backbone = model.model
    n = linears.install(backbone)
    weight = backbone._fused_kv_weight
    if weight is not None and backbone._fused_kv_bias is None:
        gemm = DraftGemm(weight)
        if gemm.eligible:
            backbone._context_kv_gemm = gemm
            n += 1
    tp = get_tp_group()
    comm = comm_module.build(model, tp)
    if comm is not None:
        if hasattr(type(backbone.fc), "comm"):
            backbone.fc.comm = comm
        model._drafter_comm = comm
        backbone._comm, backbone._tp = comm, tp
    init_logger(__name__).info_once(
        "Specialized GQA DSpark drafter: %d linears on DraftGemm, collectives %s",
        n,
        "mailbox" if comm is not None else "stock",
    )
