# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`norm/attnres_rmsnorm_common` (switch `FINALNORM`, default OFF): the model's final
attn_res and its final RMSNorm (`model.norm`) as one launch of the communicator's
`allreduce_attnres_rmsnorm` (`reduce=False`: the last layer's delta is already reduced
here; no communication).

With the switch on, `KimiLinearModel.forward` (both the decode path and the
sequence-parallel prefill path) returns hidden states that already went through
`model.norm`, and `compute_logits` / `compute_logits_local` skip it. Calls the entry
rejects (`allreduce_attnres_rmsnorm_unsupported_reason`: rows, layouts) run the stock
`attn_res` + `model.norm` inside the forward, so every returned hidden state is normed.
Numerics: the kernel normalises the fp32 mix once (the stock path rounds the mix to
bf16, then normalises): within the kernel's contract (2^-7 |y|, cosine >= 0.99999) of
the stock pair (`tests/kernels/models/amd/test_akr_nonlayer_1gpu.py::test_final_norm`).

One merge hazard: the all-reduce route owner (norm) wires the last layer's reduction
into the same kernel; with that, `reduce=True` and the un-reduced partial replace
`delta`."""

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_nonlayer

logger = init_logger(__name__)


def _reason(model) -> str | None:
    cfg = model.config
    if cfg.attn_res_block_size is None:
        return "not the Kimi-K3 attn_res model"
    for w in (
        model.norm.weight,
        model.output_attn_res_norm.weight,
        model.output_attn_res_proj.weight,
    ):
        if w.dtype != torch.bfloat16:
            return "weights are not bf16"
    return None


def setup(model) -> bool:
    """Collective (every TP rank, same order, after the weights load): the entry's
    communicator and its warm-up."""
    model._akr_final = None
    if not akr_nonlayer.switch("FINALNORM"):
        return False
    if akr_nonlayer.blocked("FINALNORM", _reason(model)):
        return False
    from amd_kernels.norm.attnres_rmsnorm_common import (
        allreduce_attnres_rmsnorm_unsupported_reason,
        attnres_comm,
        attnres_comm_unsupported_reason,
    )

    g = akr_nonlayer.tp_group()
    dev = model.norm.weight.device
    hidden = model.config.hidden_size
    kw = dict(hidden=hidden)
    if akr_nonlayer.blocked(
        "FINALNORM",
        attnres_comm_unsupported_reason(g.rank_in_group, g.cpu_group, dev, **kw),
    ):
        return False
    comm = attnres_comm(g.rank_in_group, g.cpu_group, dev, **kw)
    nb = -(-model.config.num_hidden_layers // model.config.attn_res_block_size)
    for rows in (8, 64):
        z = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=dev)
        bank = torch.zeros(rows, nb, hidden, dtype=torch.bfloat16, device=dev)
        reason = allreduce_attnres_rmsnorm_unsupported_reason(
            comm, *_args(model, z, bank, z, nb)
        )
        if akr_nonlayer.blocked("FINALNORM", reason):
            comm.close()
            return False
        model._akr_final = comm
        _launch(model, z.clone(), bank, z, nb)
    torch.accelerator.synchronize(dev)
    logger.info("amd_kernels final attn_res + final norm: on (%d bank rows)", nb)
    return True


def _args(model, prefix, bank, delta, nblocks) -> tuple:
    return (
        delta,
        prefix,
        bank,
        model.output_attn_res_norm.weight,
        model.output_attn_res_proj.weight.squeeze(0),
        nblocks,
        -1,
        model.norm.weight,
        model.output_attn_res_norm.variance_epsilon,
        model.norm.variance_epsilon,
        False,
    )


def _launch(model, prefix, bank, delta, nblocks) -> torch.Tensor:
    out, _ = model._akr_final.allreduce_attnres_rmsnorm(
        *_args(model, prefix, bank, delta, nblocks)
    )
    return out


def applied(model) -> bool:
    """Whether the model's forward returns final-normed hidden states."""
    return getattr(model, "_akr_final", None) is not None


def final_hidden(model, apply_attn_res, prefix, bank, delta, nblocks) -> torch.Tensor:
    """The forward's final step on the kernel path: the normed hidden states.
    `apply_attn_res()` is the stock `_apply_attn_res(...)` closure (a call the entry
    does not serve)."""
    from amd_kernels.norm.attnres_rmsnorm_common import (
        allreduce_attnres_rmsnorm_unsupported_reason,
    )

    if delta is not None and not akr_config.call_unsupported(
        "norm",
        allreduce_attnres_rmsnorm_unsupported_reason(
            model._akr_final, *_args(model, prefix, bank, delta, nblocks)
        ),
    ):
        return _launch(model, prefix, bank, delta, nblocks)
    return model.norm(apply_attn_res(), None)
