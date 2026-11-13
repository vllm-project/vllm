# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels ``sampling`` route (gfx950, TP4, with
``--enable-batch-sharded-sampling``): the target's logits exchange, the sampler-output
gather and the DSpark draft's Markov steps on
``amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4``. The
draft-logits cache keeps one row per request slot (``SlotDraftLogitsCache``): upstream's
compact sharded cache hands a skipped request's row to another request. Sampling is
upstream's (same draws: seed, position, vocab id; bf16 bits exchanged)."""

import functools
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.minimax_m3.amd import akr_config

logger = init_logger(__name__)

_STATE: dict[str, Any] = {}


# ---- configuration (the target's __init__)
def _blocker(vllm_config) -> str | None:
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import (
        sampling_blocker,
    )

    pc, sc = vllm_config.parallel_config, vllm_config.scheduler_config
    spec = vllm_config.speculative_config
    if not pc.enable_batch_sharded_sampling:
        return "--enable-batch-sharded-sampling is off"
    if pc.pipeline_parallel_size != 1:
        return "pipeline parallelism"
    if pc.prefill_context_parallel_size != 1 or pc.decode_context_parallel_size != 1:
        return "context parallelism"
    if getattr(vllm_config, "watermark_config", None) is not None:
        return "watermarking"
    if spec is not None and spec.method != "dspark":
        return f"speculative method {spec.method} (only the DSpark draft is wired)"
    if spec is not None and spec.enable_adaptive_verification:
        return "adaptive verification"
    steps = spec.num_speculative_tokens if spec is not None else 0
    return sampling_blocker(pc.tensor_parallel_size, sc.max_num_seqs, steps + 1)


def configure(vllm_config, model: torch.nn.Module) -> None:
    """From the target's ``__init__``: the model runner's batch-sharded hooks
    (``compute_logits_local``, ``gather_sampler_output_owned``)."""
    if not akr_config.on("sampling"):
        return
    if "blocker" not in _STATE:
        _STATE["blocker"] = _blocker(vllm_config)
        if _STATE["blocker"] is not None:
            logger.warning("amd_kernels sampling stays off: %s", _STATE["blocker"])
    if _STATE["blocker"] is not None:
        return
    spec = vllm_config.speculative_config
    steps = spec.num_speculative_tokens if spec is not None else 0
    _STATE["max_num_reqs"] = vllm_config.scheduler_config.max_num_seqs
    _STATE["target_rows"] = _STATE["max_num_reqs"] * (steps + 1)
    lm = _STATE["lm"] = getattr(model, "language_model", model)

    def compute_logits_local(hidden_states: torch.Tensor) -> torch.Tensor:
        return lm.logits_processor(lm.lm_head, hidden_states, skip_gather=True)

    model.compute_logits_local = compute_logits_local
    model.gather_sampler_output_owned = gather_sampler_output


# ---- the library block
def _sampler(device: torch.device):
    """The library's ``BatchShardedSampling``, built (and warmed up) collectively at
    first use: the draft's set-up or the target's first batch-sharded step."""
    s = _STATE.get("sampler")
    if s is None:
        from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import (  # noqa: E501
            BatchShardedSampling,
        )

        from vllm.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        s = _STATE["sampler"] = BatchShardedSampling(
            g.rank_in_group,
            g.cpu_group,
            device,
            max_reqs=int(_STATE["max_num_reqs"]),
            target_rows=int(_STATE["target_rows"]),
        )
    return s


# ---- target: logits of the owned rows
def _target_logits_owned(lm, hidden_states, sorted_logits_indices, md) -> torch.Tensor:
    """[owned logits rows, 200064] bf16 on this rank, bit for bit what upstream's
    ``compute_logits_local`` + ``all_to_all_vocab_shards`` give."""
    return _sampler(hidden_states.device).target_logits(
        hidden_states,
        lm.lm_head.weight,
        sorted_logits_indices,
        md.num_logits_per_rank,
        md.num_local_logits,
    )


# ---- target: sampler outputs back in batch order
def gather_sampler_output(
    local_output,
    metadata,
    device: torch.device,
    global_batch,
    local_batch,
    gather_num_nans: bool = False,
    logprobs_dims: tuple[int, int] | None = None,
):
    """``batch_shard.gather_sampler_output`` with the token / count / NaN part in one
    library launch; the logprobs keep upstream's gathers. Same tensors, same values."""
    from vllm.v1.worker.gpu.sample import batch_shard
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    o = local_output
    local = (
        (None,) * 5
        if o is None or metadata.num_local_reqs == 0
        else (o.sampled_token_ids, o.num_sampled, o.num_rejected, o.num_nans)
        + (local_batch.cu_num_logits,)
    )
    s = _sampler(device)
    sampled, num_sampled, num_rejected, num_nans = s.gather_sampler_output(
        *local,
        metadata.gathered_src_indices,
        metadata.max_num_reqs_per_rank,
        metadata.gathered_src_indices.shape[0],
        metadata.max_num_logits_per_req,
        gather_num_nans,
    )
    logprobs_tensors = None
    if logprobs_dims is not None:
        logprobs_tensors = batch_shard._gather_logprobs_tensors(
            local_output, metadata, global_batch, local_batch, logprobs_dims, device
        )
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs_tensors,
        num_nans=num_nans,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


# ---- the draft-logits cache
def _slot_cache(compact, dtype: torch.dtype, fill: float):
    from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

    class SlotDraftLogitsCache(DraftLogitsCache):
        """Batch-sharded, with the replicated cache's rows: row = request slot (identity
        ``row_mapping``), written by the rank that drafts and then verifies the request
        (batch position % 4); a request the scheduler skips keeps its row."""

        def __init__(self) -> None:  # noqa: D107
            # not DraftLogitsCache.__init__: the owner tables carry over
            num_slots = compact.shard_mapping_np.shape[0]
            steps, vocab = compact.values.shape[1:]
            dev = compact.values.device
            self._num_shards = compact._num_shards
            self._shard_idx = compact._shard_idx
            self.shard_mapping_np = compact.shard_mapping_np
            self.shard_mapping = compact.shard_mapping
            self.row_mapping = torch.arange(num_slots, dtype=torch.int32, device=dev)
            self.inverse_row_mapping = self.row_mapping.clone()
            self._pos_owner = self.row_mapping % self._num_shards
            compact.values = None  # free the compact rows before the slot rows
            torch.accelerator.empty_cache()
            self.values = torch.full(
                (num_slots, steps, vocab), fill, dtype=dtype, device=dev
            )

        def assign(self, idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray) -> None:
            """Batch position i drafts on rank i % 4, which verifies it next."""
            n = idx_mapping_np.shape[0]
            if n == 0:
                return
            self.shard_mapping_np[idx_mapping_np] = np.arange(n) % self._num_shards
            self.shard_mapping[idx_mapping] = self._pos_owner[:n]

    return SlotDraftLogitsCache()


# ---- the draft's set-up and its batch-sharded Markov loop
def setup(spec, model: torch.nn.Module, target_model: torch.nn.Module) -> None:
    """From the draft's set-up (every TP rank, before any capture): the target's owned
    logits on the shuffled lm_head, the slot-row draft cache and, where the draft fits,
    the library's Markov chain (collective; it warms up on its first call)."""
    from amd_kernels.sampling.batch_sharded_markov_v200064_h6144_r256_ranks4 import (
        markov_blocker,
    )

    from vllm.models.minimax_m3.amd.dspark_akr import plain_processor

    lm = _STATE.get("lm")
    if lm is None:
        return
    s = _sampler(spec.device)  # collective, before any capture
    if getattr(lm.lm_head, "_akr_shuffled", False) and plain_processor(
        lm.logits_processor
    ):
        target_model.compute_logits_owned = functools.partial(_target_logits_owned, lm)
    dl = spec.draft_logits
    if dl is None or dl.row_mapping is None:
        logger.info("amd_kernels sampling: the draft samples stock (no sharded cache)")
        return
    dtype, fill = spec.draft_logits_spec(spec.vllm_config)
    spec.draft_logits = _slot_cache(dl, dtype, fill)
    head = model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    reason = markov_blocker(
        spec.num_speculative_steps,
        spec.max_num_reqs,
        w1,
        w2,
        cache=spec.draft_logits.values,
        lm_weight_shuffled=getattr(model.lm_head, "_akr_shuffled", False),
        fp64_noise=spec.use_fp64_gumbel,
        confidences=spec.use_confidence_head,
        logit_transform=hasattr(model, "apply_output_logit_transform"),
        reduced_vocab=spec._d2t_scatter_index is not None,
    )
    if reason is None:
        s.setup_markov(
            spec.num_speculative_steps, spec.max_num_reqs, spec._anchor_idx, w1, w2
        )
        model.sample_sequential_sharded = functools.partial(_markov_draft, model)
    torch.accelerator.synchronize(spec.device)
    logger.info(
        "amd_kernels sampling: the draft's Markov loop %s (slot-row draft cache)",
        "runs amd_kernels" if reason is None else f"stays stock: {reason}",
    )


def _markov_draft(model, spec, num_reqs: int, head_hidden: torch.Tensor) -> None:
    """The speculator's ``sample_sequential_sharded`` hook: its Markov loop on the
    library chain (same draws, ids and cache rows)."""
    _STATE["sampler"].markov_draft(
        num_reqs,
        head_hidden,
        model.lm_head.weight,
        spec.sample_indices,
        spec.sample_idx_mapping,
        spec.sample_pos,
        spec.input_buffers.input_ids,
        spec.temperature,
        spec.seeds,
        spec.draft_logits.values,
        spec.draft_tokens,
    )
