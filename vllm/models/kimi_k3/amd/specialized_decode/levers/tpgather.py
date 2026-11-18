# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers on TP all-gathers around the DSpark draft (ROCm).

``VLLM_K31_BEAT_DRAFTFC``: a DFlash-family draft (the Kimi-K3.1 release DSpark)
projects the target's concatenated aux hidden states (5 x 7168 for Kimi-K3.1)
to its hidden size with ``fc``, a ``ReplicatedLinear``: every rank reads the
whole ~0.5 GB bf16 weight each step (~87 us on MI355X at c1). Here each rank
keeps the weight's rows of its output columns (``1 / tp`` of them) and the
outputs are all-gathered along the last dim. Every output element is the same
dot product as before (only the GEMM's tiling changes), and each rank frees
the rest of the weight.

``VLLM_K31_BEAT_LOGITSAG``: ``LogitsProcessor``'s vocab-parallel all-gather
(the target's and the draft's lm_head) runs over the TP group's AITER IPC
all-gather instead of RCCL (~23 us a call on MI355X at c1).

``VLLM_K31_BEAT_MARKOVTP``: the fused Markov launches (``levers.dspark``, one
request) split the vocabulary over TP: each rank reads 1 / tp of the 168 MB
``markov_w2`` a step (~27 us whole on MI355X), the step maxima are all-gathered
between the launches and the draft logits once after them
(``markov_sample.run``'s ``tp``). Every token and cached logit is the
whole-vocabulary launch's.

Both gathers take the AITER all-gather when the payload fits its IPC buffer
(a choice that reads only the shape, the dtype and the group, so every rank
makes the same one) and RCCL's otherwise (prefill-sized inputs)."""

import functools

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd.specialized_decode.levers import (
    DRAFTFC,
    LOGITSAG,
    MARKOVTP,
    lever_on,
    wrap,
)

logger = init_logger(__name__)

# The AITER all-gather copies 16-byte packs.
_PACK_BYTES = 16


def install() -> None:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    wrap(LogitsProcessor, "_gather_logits", _wrap_gather_logits)
    # DSpark overrides DFlash's loader; shard_draft_fc runs once either way.
    wrap(DFlashSpeculator, "load_draft_model", _wrap_load_draft_model)
    wrap(DSparkSpeculator, "load_draft_model", _wrap_load_draft_model)


def _tp_aiter_comm():
    from vllm.distributed import get_tp_group

    tp = get_tp_group()
    if tp.world_size == 1:
        return None
    comm = getattr(tp.device_communicator, "aiter_ar_comm", None)
    if comm is None or comm.disabled:
        return None
    return comm


def aiter_all_gather_last(x: torch.Tensor) -> torch.Tensor | None:
    """The last-dim TP all-gather of ``x`` over the AITER IPC all-gather, or
    None when it cannot take it (no AITER comm, rows not 16-byte packs, a
    payload larger than its buffer holds)."""
    comm = _tp_aiter_comm()
    if comm is None or x.numel() == 0:
        return None
    row_bytes = x.shape[-1] * x.element_size()
    if row_bytes % _PACK_BYTES:
        return None
    # Copy an input the kernel cannot read in place (never fall back on it:
    # the choice would depend on the allocator and ranks could split).
    if not x.is_contiguous():
        x = x.contiguous()
    if x.data_ptr() % _PACK_BYTES:
        x = x.clone()
    payload = x.view(-1, x.shape[-1]).view(torch.int32)
    if not comm.should_custom_ag(payload):
        return None
    out = comm.custom_all_gather(payload, dim=1)
    if out is None:
        return None
    return out.view(x.dtype).view(*x.shape[:-1], -1)


def all_gather_last(x: torch.Tensor) -> torch.Tensor:
    out = aiter_all_gather_last(x)
    if out is not None:
        return out
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x, -1)


def markov_tp(num_reqs: int, vocab: int):
    """``VLLM_K31_BEAT_MARKOVTP``: ``markov_sample.run``'s ``tp`` (this rank,
    the TP size, the last-dim all-gather), or None when the fused Markov
    launches keep the whole vocabulary on every rank."""
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.models.kimi_k3.amd.ops import markov_sample

    if not (lever_on(MARKOVTP) and model_parallel_is_initialized()):
        return None
    tp = get_tp_group()
    why = markov_sample.tp_split_why_not(num_reqs, vocab, tp.world_size)
    if why is None and _tp_aiter_comm() is None:
        why = "no AITER all-gather"
    if why is not None:
        logger.info_once("DSpark Markov TP split off: %s", why)
        return None
    logger.info_once(
        "DSpark Markov TP split: %d vocabulary tokens a rank, the step maxima "
        "and the logits all-gathered",
        vocab // tp.world_size,
    )
    return tp.rank_in_group, tp.world_size, all_gather_last


def _wrap_gather_logits(stock):
    @functools.wraps(stock)
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if lever_on(LOGITSAG) and self.use_all_gather:
            out = aiter_all_gather_last(logits)
            if out is not None:
                return out
        return stock(self, logits)

    return _gather_logits


class TPShardedFC(nn.Module):
    """``fc`` with this rank's output columns, all-gathered along the last dim
    (``ReplicatedLinear``'s interface as ``combine_hidden_states`` reads it)."""

    def __init__(self, fc: nn.Module, tp_rank: int, tp_size: int) -> None:
        super().__init__()
        n = fc.output_size // tp_size
        self.input_size = fc.input_size
        self.output_size = fc.output_size
        self.params_dtype = fc.params_dtype
        # fc's UnquantizedLinearMethod: its GEMM dispatch, on this weight.
        self.quant_method = fc.quant_method
        self.weight = nn.Parameter(
            fc.weight.data[tp_rank * n : (tp_rank + 1) * n].clone(),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return all_gather_last(self.quant_method.apply(self, x))


def draft_fc_why_not(model: nn.Module) -> str | None:
    """Why ``VLLM_K31_BEAT_DRAFTFC`` leaves this draft's ``fc`` replicated."""
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )

    inner = getattr(model, "model", None)
    fc = getattr(inner, "fc", None)
    if not getattr(inner, "use_aux_hidden_state", False) or fc is None:
        return "the draft has no aux-hidden fc"
    if not isinstance(fc, ReplicatedLinear):
        return f"its fc is a {type(fc).__name__}"
    if not isinstance(fc.quant_method, UnquantizedLinearMethod):
        return "its fc is quantized"
    if fc.bias is not None:
        return "its fc has a bias"
    tp = get_tensor_model_parallel_world_size()
    if tp == 1:
        return "TP 1"
    if fc.output_size % tp:
        return f"fc's {fc.output_size} outputs do not split over TP {tp}"
    if fc.weight.dtype not in (torch.bfloat16, torch.float16):
        return f"fc's weight is {fc.weight.dtype}"
    return None


def shard_draft_fc(model: nn.Module) -> bool:
    """Swaps the draft's replicated ``fc`` for ``TPShardedFC`` (once)."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    if isinstance(getattr(getattr(model, "model", None), "fc", None), TPShardedFC):
        return True
    why = draft_fc_why_not(model)
    if why is not None:
        logger.info_once("Draft fc TP sharding off: %s", why)
        return False
    fc = model.model.fc
    tp = get_tensor_model_parallel_world_size()
    model.model.fc = TPShardedFC(fc, get_tensor_model_parallel_rank(), tp)
    del fc
    logger.info_once(
        "Draft fc TP sharding: %d x %d, %d output columns a rank, all-gathered",
        model.model.fc.output_size, model.model.fc.input_size,
        model.model.fc.output_size // tp,
    )  # fmt: skip
    return True


def _wrap_load_draft_model(stock):
    @functools.wraps(stock)
    def load_draft_model(self, target_model, target_attn_layer_names):
        model = stock(self, target_model, target_attn_layer_names)
        if lever_on(DRAFTFC):
            shard_draft_fc(model)
        return model

    return load_draft_model
