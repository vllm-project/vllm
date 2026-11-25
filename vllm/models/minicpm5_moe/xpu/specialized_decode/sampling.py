# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in TP2 greedy vocabulary candidates with stable global token IDs."""

import numpy as np
import torch
import torch.distributed as dist


def candidate_head_supported(head, processor, rank: int) -> bool:
    shard_size = 65_280
    return (
        head.tp_size == 2
        and head.num_embeddings_per_partition == shard_size
        and head.num_embeddings_padded == head.org_vocab_size == 2 * shard_size
        and head.shard_indices.org_vocab_start_index == rank * shard_size
        and processor.org_vocab_size == head.org_vocab_size
        and processor.head_dtype in (None, torch.float16, torch.bfloat16)
        and not processor.logits_as_input
    )


def request_ok(params):
    if (
        params.logprobs is not None
        or getattr(params, "logprob_token_ids", None)
        or params.presence_penalty != 0.0
        or params.frequency_penalty != 0.0
        or params.repetition_penalty != 1.0
        or params.logit_bias
        or params.bad_words
        or params.allowed_token_ids
        or params.min_tokens
        or params.min_p != 0.0
        or params.thinking_token_budget is not None
        or params.structured_outputs is not None
    ):
        return False
    # Stock top-k retains every threshold tie. A fixed K candidate list can
    # truncate BF16 ties, so random requests retain the full vocabulary.
    return params.temperature == 0.0


def enabled(config):
    extra = config.additional_config
    if not isinstance(extra, dict) or not extra.get("minicpm5_candidate_sampling"):
        return False
    model = config.model_config
    return (
        config.parallel_config.tensor_parallel_size == 2
        and config.speculative_config is None
        and not model.logits_processors
        and not model.masked_token_ids
        and config.watermark_config is None
        and not model.return_sampling_mask
        and not model.enable_trace_replay
    )


class Candidates:
    def __init__(self, config, decode, max_requests, head, processor):
        from vllm import envs

        from .peer import agree, region

        self.enabled = enabled(config) and not envs.VLLM_COMPUTE_NANS_IN_LOGITS
        self.ok = np.zeros(max_requests, dtype=bool)
        self.batch = None
        self.indices = None
        self.hits = 0
        self.direct_hits = 0
        self.direct_sampler = None
        self.direct_step = False
        self.direct_batch = None
        self.direct_values = None
        self.direct_outputs = None
        self.ops = decode.ops
        self.rank = decode.tp.rank_in_group
        self.region = None
        if not self.enabled:
            return
        compatible = torch.tensor(
            int(candidate_head_supported(head, processor, self.rank)),
            dtype=torch.int32,
            device="cpu",
        )
        dist.all_reduce(compatible, op=dist.ReduceOp.MIN, group=decode.tp.cpu_group)
        self.enabled = bool(compatible.item())
        if not self.enabled:
            return
        agree(
            []
            if hasattr(self.ops, "candidate_topk")
            else ["Rebuild native candidate sampling"],
            decode.tp.cpu_group,
            "candidate deployment",
        )
        self.region = region(1, decode.tp, self.ops)
        device = next(decode.model.parameters()).device
        self.partial = torch.empty((8, 32, 64), dtype=torch.int64, device=device)
        self.control = torch.zeros(
            self.ops.control_words(), dtype=torch.int32, device=device
        )
        if decode.failstop is None:
            decode.add_health_control(self.control)
        else:
            self.region.set_health(decode.region.health())

    def add_request(self, index, params):
        self.ok[index] = request_ok(params)

    def plan(self, rows, grammar_output):
        batch, self.batch, self.indices = self.batch, None, None
        self.direct_step = False
        self.direct_batch = self.direct_values = self.direct_outputs = None
        if (
            not self.enabled
            or batch is None
            or grammar_output is not None
            or batch.has_prefill
            or batch.has_structured_output_reqs
            or batch.num_draft_tokens
            or not 1 <= rows <= 8
            or rows != batch.num_reqs
            or not (batch.num_scheduled_tokens == 1).all()
        ):
            return None
        requests = batch.idx_mapping_np
        if np.any(requests < 0) or not np.all(self.ok[requests]):
            return None
        sampler = getattr(self, "direct_sampler", None)
        if (
            sampler is not None
            and not np.any(batch.is_prefilling_np)
            and np.all(np.diff(batch.cu_num_logits_np) == 1)
            and not np.any(sampler.needs_logits_processing[requests])
            and sampler.get_logprobs_dims(requests) is None
        ):
            self.direct_step = True
            self.direct_batch = batch
        return 1

    def logits(self, shard, k):
        values = torch.empty(
            (shard.shape[0], 128), dtype=torch.float32, device=shard.device
        )
        indices = torch.empty_like(values, dtype=torch.int32)
        if self.direct_step:
            rows, device = shard.shape[0], shard.device
            sampled = torch.empty((rows, 1), dtype=torch.int64, device=device)
            counts = torch.empty(rows, dtype=torch.int32, device=device)
            rejected = torch.empty_like(counts)
            self.ops.candidate_greedy(
                values,
                indices,
                shard.contiguous(),
                self.rank,
                self.partial,
                self.control,
                self.region,
                sampled,
                counts,
                rejected,
            )
            self.direct_values = values
            self.direct_outputs = sampled, counts, rejected
            self.indices = indices
            self.direct_hits += 1
        else:
            self.ops.candidate_topk(
                values,
                indices,
                shard.contiguous(),
                k,
                self.rank,
                self.partial,
                self.control,
                self.region,
            )
            self.indices = indices.to(torch.int64)
        self.hits += 1
        return values

    def take(self, logits):
        indices, self.indices = self.indices, None
        if indices is None:
            return None
        if indices.shape != logits.shape:
            raise RuntimeError("MiniCPM candidate logits and token IDs disagree")
        return indices


class CandidateSampler:
    def __init__(self, sampler, candidates):
        self.sampler, self.candidates = sampler, candidates
        from vllm.v1.worker.gpu.sample.sampler import Sampler

        ops = getattr(candidates, "ops", None)
        greedy_version = getattr(ops, "candidate_greedy_version", None)
        if (
            type(sampler) is Sampler
            and greedy_version is not None
            and greedy_version() == 1
            and not sampler.compute_nans
            and not sampler.return_sampling_mask
            and not sampler.use_fp64_gumbel
            and sampler.trace_replay_state is None
        ):
            candidates.direct_sampler = sampler

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def add_request(self, index, params):
        self.sampler.add_request(index, params)
        self.candidates.add_request(index, params)

    def __call__(self, logits, input_batch):
        c = self.candidates
        direct = getattr(c, "direct_outputs", None)
        c.direct_outputs = None
        if (
            direct is not None
            and logits is c.direct_values
            and input_batch is c.direct_batch
        ):
            from vllm.v1.worker.gpu.sample.output import SamplerOutput

            c.indices = c.direct_values = c.direct_batch = None
            sampled, counts, rejected = direct
            return SamplerOutput(
                sampled_token_ids=sampled,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=counts,
                num_rejected=rejected,
                sampling_mask_tensors=None,
            )
        c.direct_values = c.direct_batch = None
        indices = self.candidates.take(logits)
        output = self.sampler(logits, input_batch)
        if indices is not None:
            tokens = output.sampled_token_ids
            output.sampled_token_ids = (
                indices.to(torch.int64)
                .gather(1, tokens.to(torch.int64))
                .to(tokens.dtype)
            )
        return output
