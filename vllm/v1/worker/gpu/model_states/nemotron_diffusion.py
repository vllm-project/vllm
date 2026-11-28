# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker state and sampling for Nemotron masked block diffusion."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.config import NemotronDiffusionOptions
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.utils.torch_version import is_torch_equal_or_newer
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.logprob import (
    compute_topk_scores,
    concat_logprob_stashes,
    masked_stash_write,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty
from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

# torch < 2.14 Inductor can reuse an indirect load from inside a reduction loop
# after the loop closes (pytorch#194490). Drop once ROCm uses torch >= 2.14.
_INDUCTOR_GATHER_OK = is_torch_equal_or_newer("2.14.0")


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


def _filter_sampling_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    mask_token_id: int,
) -> torch.Tensor:
    """Select sampling candidates after temperature scaling, excluding masks."""
    logits = logits.float().clone()
    logits[:, mask_token_id] = float("-inf")
    temperatures = torch.where(temperatures > 0, temperatures, 1.0)
    filtered = apply_top_k_top_p(logits / temperatures[:, None], top_k, top_p)
    # Retain unscaled logits for the confidence-based reveal policy.
    return logits.masked_fill_(filtered.isneginf(), float("-inf"))


@torch.compile(dynamic=True)
def _compiled_masked_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,  # [num_decode] int64 -> slot indices
    decode_idx: torch.Tensor,  # [num_decode] int64 -> position in num_reqs
    all_slots: torch.Tensor,  # [num_reqs] int64 -> all slot indices
    valid_canvas_len: torch.Tensor,  # [num_decode] int64 real canvas len (<=CL)
    req_temps: torch.Tensor,  # [num_decode] fp32 per-request temperature (0=greedy)
    # State tensors (modified in-place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs]
    unmask_logprobs: torch.Tensor,  # [max_num_reqs, CL] fp32
    unmask_ranks: torch.Tensor,  # [max_num_reqs, CL] int32
    # Output tensors (modified in-place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: int,
    mask_token_id: int,
    CL: int,
    all_greedy: bool,
    capture_logprobs: bool,
    leftmost: bool,
    threshold_mode: bool,
    log_threshold: float,
) -> torch.Tensor:
    """Update absorbing masked-token state and emit completed blocks.

    Confidence uses the selected token's probability before temperature scaling.
    Capture token logprobs when positions are revealed, then refresh causal KV
    on the following commit step. Return logits for top-k introspection.
    """
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    logits_3d = logits.reshape(num_decode, CL, -1).float().clone()
    logits_3d[..., mask_token_id] = float("-inf")

    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit

    cur_canvas = canvas[decode_slots]  # [num_decode, CL]
    pos = torch.arange(CL, device=device)
    # A canvas truncated near max_model_len is zero-padded up to CL by the
    # caller; the padded tail is treated as non-masked so it is never
    # transferred or committed.
    valid = pos.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
    mask_index = (cur_canvas == mask_token_id) & valid

    # ---- x0 + confidence (prob of chosen token) at masked positions ----
    if not all_greedy:
        # Gumbel-max: argmax(logits + T*g) ~ softmax(logits / T).
        # Rows whose request asked for greedy (per-request temperature 0)
        # get zero noise and reduce to argmax.
        u = torch.rand_like(logits_3d).clamp_min(1e-20)
        gumbel = -torch.log(-torch.log(u))
        x0 = (logits_3d + req_temps[:, None, None] * gumbel).argmax(dim=-1)
    else:
        x0 = logits_3d.argmax(dim=-1)  # [num_decode, CL]
    if _INDUCTOR_GATHER_OK:
        chosen_logits = logits_3d.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        # Masked max: same value as the gather, without the indirect load.
        vocab_ids = torch.arange(logits_3d.shape[-1], device=device)
        chosen_logits = torch.where(
            vocab_ids == x0.unsqueeze(-1), logits_3d, float("-inf")
        ).amax(dim=-1)
    x0_logprob = chosen_logits - logits_3d.logsumexp(dim=-1)
    neg_inf = torch.full_like(x0_logprob, float("-inf"))
    confidence = torch.where(mask_index, x0_logprob, neg_inf)

    # ---- Even transfer schedule: k = ceil(#masks / remaining steps) ----
    num_masks = mask_index.sum(dim=-1)  # [num_decode] int64
    cur_step = step_tensor[decode_slots].to(torch.int64)
    remaining_steps = (max_denoising_steps - cur_step).clamp(min=1)
    k = -(-num_masks // remaining_steps)  # ceil div

    if leftmost:
        # Leftmost-reveal: unmask the first k masked positions (strict
        # left-to-right order instead of confidence order).
        csum = torch.cumsum(mask_index.to(torch.int64), dim=-1)
        transfer = mask_index & (csum <= k.unsqueeze(1))
    elif threshold_mode:
        # FastDiffuser thresholding: unmask every masked position whose
        # chosen-token logprob clears the threshold, and always the single
        # most confident one (guaranteed progress; also covers the case
        # where nothing clears the bar). `confidence` is -inf outside
        # mask_index, so argmax lands on a masked position whenever one
        # exists and both terms are automatically gated by mask_index.
        above = confidence >= log_threshold
        top1 = confidence.argmax(dim=-1, keepdim=True)  # [num_decode, 1]
        force = torch.zeros_like(mask_index).scatter_(1, top1, True)
        transfer = mask_index & (above | force | (remaining_steps == 1).unsqueeze(1))
    else:
        # Top-k per row with per-row k: rank positions by confidence (desc).
        order = torch.argsort(confidence, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, pos.expand(num_decode, CL))
        transfer = mask_index & (ranks < k.unsqueeze(1))

    # ---- At-unmask logprob/rank capture (sampling distribution) ----
    sample_logits = logits_3d
    if capture_logprobs:
        # Greedy requests report unscaled model logprobs, as in AR sampling.
        row_temp = torch.where(req_temps > 0, req_temps, 1.0)
        sample_logits = logits_3d / row_temp[:, None, None]
        if _INDUCTOR_GATHER_OK:
            chosen_sample = sample_logits.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            chosen_sample = chosen_logits / row_temp[:, None]
        chosen_lp = chosen_sample - sample_logits.logsumexp(dim=-1)
        chosen_rank = (
            (logits_3d >= chosen_logits.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
        )
        write = transfer & is_denoise.unsqueeze(1)
        unmask_logprobs[decode_slots] = torch.where(
            write, chosen_lp, unmask_logprobs[decode_slots]
        )
        unmask_ranks[decode_slots] = torch.where(
            write, chosen_rank, unmask_ranks[decode_slots]
        )

    # ---- Absorbing-state update (denoise rows only) ----
    new_canvas = torch.where(transfer & is_denoise.unsqueeze(1), x0, cur_canvas)

    # Commit rows: reset the canvas to masks for the next block.
    mask_canvas = torch.full_like(cur_canvas, mask_token_id)
    canvas[decode_slots] = torch.where(is_commit.unsqueeze(1), mask_canvas, new_canvas)

    # Step counter: +1 for denoise, reset to 0 for commit.
    new_step = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step

    # Sampled output: the commit step emits the canvas that was fed to the
    # model this step; denoise steps emit nothing. A canvas truncated by
    # max_model_len can converge with unresolved masks in its tail (only the
    # scheduled prefix is denoised); never emit mask tokens — commit stops at
    # the first remaining mask (for leftmost reveal this is exactly the clean
    # prefix).
    mask_in_canvas = (cur_canvas == mask_token_id) & valid
    first_mask = torch.where(
        mask_in_canvas.any(dim=-1),
        mask_in_canvas.to(torch.int64).argmax(dim=-1),
        valid_canvas_len,
    )
    emit_len = torch.minimum(valid_canvas_len, first_mask)
    sampled[decode_idx] = cur_canvas.to(sampled.dtype) * is_commit.unsqueeze(1).to(
        sampled.dtype
    )
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * emit_len.to(
        num_sampled.dtype
    )

    # ---- Convergence: no masks remain, or step budget exhausted ----
    masks_left = ((new_canvas == mask_token_id) & valid).sum(dim=-1)
    converged = (masks_left == 0) | (new_step.to(torch.int64) >= max_denoising_steps)
    # Commit done -> denoise next (False); denoise converged -> commit (True).
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # Feed the canvas back to the scheduler as next step's draft tokens.
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return sample_logits


class NemotronLabsDiffusionRequestStates:
    """Pre-allocated GPU tensors for per-request masked-diffusion state."""

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        mask_token_id: int,
        max_denoising_steps: int,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.max_denoising_steps = max_denoising_steps
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length], mask_token_id when
        # not yet unmasked.
        self.canvas = torch.full(
            (max_num_reqs, canvas_length),
            mask_token_id,
            dtype=torch.int64,
            device=device,
        )
        # Denoising step counter (0..max_denoising_steps).
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Per-position logprob/rank of the chosen token, captured at the
        # denoise step each position was unmasked; emitted at commit.
        self.unmask_logprobs = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.float32, device=device
        )
        self.unmask_ranks = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int32, device=device
        )

    def init_canvas(self, slot_indices) -> None:
        """Reset the canvas to all-mask for the given slots."""
        self.canvas.index_fill_(0, slot_indices, self.mask_token_id)

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx].fill_(True)
        self.canvas[slot_idx] = self.mask_token_id
        self.step[slot_idx].fill_(0)
        self.unmask_logprobs[slot_idx] = 0.0
        self.unmask_ranks[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx].fill_(False)


class NemotronLabsDiffusionModelState(ModelState):
    """ModelState for the Ministral masked block-diffusion LM.

    Single backbone in two modes:
    - encoder mode (prefill/commit): causal attention, writes KV
    - denoise mode: bidirectional attention over the canvas
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        hf_config = self.model_config.hf_config
        diffusion_config = vllm_config.diffusion_config
        assert diffusion_config is not None
        options = diffusion_config.model_options
        assert isinstance(options, NemotronDiffusionOptions)
        canvas_length = diffusion_config.canvas_length
        gen_config = self.model_config.try_get_generation_config()
        # Default to one unmasked token per step (HF steps == gen_length).
        max_denoising_steps = (
            diffusion_config.max_denoising_steps
            or gen_config.get("max_denoising_steps")
            or canvas_length
        )
        selection = (
            options.selection_policy
            or gen_config.get("diffusion_selection_policy")
            or "confidence_threshold"
        )
        if selection not in ("low_confidence", "leftmost", "confidence_threshold"):
            raise ValueError(f"Unknown diffusion selection_policy: {selection!r}")
        self.selection_policy = selection
        # Probability threshold for selection_policy="confidence_threshold"
        # (SGLang FastDiffuser thresholding). None-checks rather than `or`
        # so an explicit small value is honored.
        threshold = options.confidence_threshold
        if threshold is None:
            threshold = gen_config.get("diffusion_confidence_threshold")
        if threshold is None:
            threshold = 0.9
        self.confidence_threshold = float(threshold)
        self.diffusion_states = NemotronLabsDiffusionRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            mask_token_id=getattr(hf_config, "mask_token_id", 100),
            max_denoising_steps=max_denoising_steps,
            device=device,
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        return MaskedDiffusionSampler(
            sampler=sampler,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            selection_policy=self.selection_policy,
            confidence_threshold=self.confidence_threshold,
            max_logprobs=self.model_config.max_logprobs,
        ), None

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        # Text-only model, no self-conditioning: use the default
        # input_ids path.
        return {}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {}

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Per-request causal mode: encoder (prefill/commit) = causal,
        # denoise = bidirectional. Pass a GPU tensor so the attention
        # backend can handle mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        # Invariant: the sampler flips is_encoder_phase to False only after a
        # request's FINAL prompt chunk, so a prompt spanning multiple chunks
        # stays causal for every chunk.
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=causal,
        )

    num_new_sampled_tokens_per_step: int = 0


# Penalty stub for the diffusion path: the runner reads
# penalties_state.output_bin_counts, and post_update treats None as
# "no penalty bookkeeping".
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class MaskedDiffusionSampler:
    """Masked diffusion with per-request sampling temperatures.

    The default threshold policy reveals all confident positions and guarantees
    at least one reveal per step. `logprobs=0` returns each token's logprob at
    reveal time; top-k introspection uses the converged denoising distribution.
    """

    def __init__(
        self,
        sampler: Any,
        vocab_size: int,
        diffusion_states: NemotronLabsDiffusionRequestStates,
        max_logprobs: int,
        selection_policy: str = "confidence_threshold",
        confidence_threshold: float = 0.9,
    ):
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.canvas_length = diffusion_states.canvas_length
        self.mask_token_id = diffusion_states.mask_token_id
        self.leftmost = selection_policy == "leftmost"
        self.threshold_mode = selection_policy == "confidence_threshold"
        # Compared against per-position chosen-token logprobs (natural log).
        self.log_threshold = math.log(confidence_threshold)

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs, self.canvas_length, dtype=torch.int32, device=device
        )
        self._num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # Per-slot top-k (logprobs>0) of the converging denoise step, read on
        # the following commit step. Fixed width, allocated before memory
        # profiling.
        if max_logprobs < 0 or max_logprobs > MAX_LOGPROB_TOKEN_IDS:
            max_logprobs = MAX_LOGPROB_TOKEN_IDS
        self._stash_width = max_logprobs
        self._topk_stash: LogprobsTensors | None = None
        if max_logprobs > 0:
            shape = (max_num_reqs, self.canvas_length, 1 + max_logprobs)
            self._topk_stash = LogprobsTensors(
                logprob_token_ids=torch.zeros(shape, dtype=torch.int64, device=device),
                logprobs=torch.full(shape, float("-inf"), device=device),
                selected_token_ranks=torch.zeros(
                    shape[:2], dtype=torch.int64, device=device
                ),
            )
        # Requests with a wider top-k, filled on the step a slot converges:
        # a [canvas_length, vocab] stash per slot would not fit.
        self._wide_logprobs: dict[int, LogprobsTensors] = {}

    def add_request(self, req_idx: int, sampling_params: Any) -> None:
        self._wide_logprobs.pop(req_idx, None)
        if use_penalty(sampling_params):
            logger.warning_once(
                "Masked block-diffusion does not support repetition/frequency/"
                "presence penalties; ignoring them for this request."
            )
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        return _NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Transition requests whose prompt completes this step to denoising.

        Initializes their all-mask canvas, seeds draft tokens, and flips
        is_encoder_phase to False. Mid-chunk requests (prompt longer than the
        token budget) are left untouched so is_encoder_phase stays True and
        prepare_attn keeps causal attention for their remaining chunks.
        """
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        ps_gpu = async_tensor_h2d(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        states.init_canvas(ps_gpu)
        self.req_states.draft_tokens[ps_gpu, : self.canvas_length] = states.canvas[
            ps_gpu
        ]
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(self, input_batch: Any) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _stash_topk_logprobs(
        self,
        logits_3d: torch.Tensor,
        canvas_rows: torch.Tensor,
        slots: torch.Tensor,
        keep: torch.Tensor,
        num_logprobs: int,
    ) -> None:
        """Stash the top-k of `slots` where `keep` (the step they converged).

        `logits_3d` is [n, CL, vocab] and `canvas_rows` the converged canvases
        [n, CL]. Rows past a truncated canvas are never emitted.
        """
        assert self._topk_stash is not None
        n, CL, vocab = logits_3d.shape
        new = compute_topk_scores(
            logits_3d.reshape(n * CL, vocab), num_logprobs, canvas_rows.reshape(-1)
        )
        masked_stash_write(self._topk_stash, slots, keep, new)

    def _pend_wide_logprobs(
        self,
        logits_3d: torch.Tensor,
        canvas_rows: torch.Tensor,
        wide_rows_np: np.ndarray,
        slots_np: np.ndarray,
        valid_len_np: np.ndarray,
        keep: torch.Tensor,
        num_logprobs: int,
    ) -> None:
        """Compute the top-k of the wide decode rows that converged this step.

        A per-slot stash for them would hold [CL, vocab] rows, and a top-k
        over all their rows every step is a full sort. So read back which of
        them converged and compute only those.
        """
        with gpu_sync_allowed():
            converged_np = keep.cpu().numpy()
        for i in wide_rows_np[converged_np[wide_rows_np]].tolist():
            k = int(valid_len_np[i])
            self._wide_logprobs[int(slots_np[i])] = compute_topk_scores(
                logits_3d[i, :k], num_logprobs, canvas_rows[i, :k]
            )

    def _canvas_rows(
        self,
        sampled: torch.Tensor,
        req_idx_np: np.ndarray,
        slots_np: np.ndarray,
        use_topk_np: np.ndarray,
        width: int,
    ) -> LogprobsTensors:
        """[n, CL, width] logprobs rows for the requests at `req_idx_np`.

        logprobs=0 requests report the at-unmask logprob; top-k requests read
        the stash written on their converging step.
        """
        states = self.diffusion_states
        CL = self.canvas_length
        device = sampled.device
        n = len(req_idx_np)
        req_idx = async_tensor_h2d(req_idx_np.astype(np.int64), device=device)
        slots = async_tensor_h2d(slots_np.astype(np.int64), device=device)

        ids = torch.zeros(n, CL, width, dtype=torch.int64, device=device)
        ids[..., 0] = sampled[req_idx].to(torch.int64)
        lps = torch.full((n, CL, width), float("-inf"), device=device)
        lps[..., 0] = states.unmask_logprobs[slots]
        ranks = states.unmask_ranks[slots].to(torch.int64)
        stash = self._topk_stash
        if stash is not None and use_topk_np.any():
            use_topk = async_tensor_h2d(use_topk_np, device=device)
            ids = torch.where(
                use_topk.view(n, 1, 1), stash.logprob_token_ids[slots, :, :width], ids
            )
            lps = torch.where(
                use_topk.view(n, 1, 1), stash.logprobs[slots, :, :width], lps
            )
            ranks = torch.where(
                use_topk.view(n, 1), stash.selected_token_ranks[slots], ranks
            )
        return LogprobsTensors(
            logprob_token_ids=ids, logprobs=lps, selected_token_ranks=ranks
        )

    def _canvas_logprobs(
        self,
        num_reqs: int,
        sampled: torch.Tensor,
        req_idx_np: np.ndarray,
        slots_np: np.ndarray,
        use_topk_np: np.ndarray,
        width: int,
    ) -> LogprobsTensors:
        """A canvas of logprobs rows for every logprobs request in decode.

        Only committing requests read their rows (the scheduler slices
        num_sampled of them), so which requests commit never has to be known
        on the host.
        """
        rows = self._canvas_rows(sampled, req_idx_np, slots_np, use_topk_np, width)
        rows_np = np.zeros(num_reqs, dtype=np.int64)
        rows_np[req_idx_np] = self.canvas_length
        return LogprobsTensors(
            logprob_token_ids=rows.logprob_token_ids.view(-1, width),
            logprobs=rows.logprobs.view(-1, width),
            selected_token_ranks=rows.selected_token_ranks.view(-1),
            cu_num_generated_tokens=(np.cumsum(rows_np) - rows_np).tolist(),
        )

    def _emitted_logprobs(
        self,
        num_reqs: int,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        req_idx_np: np.ndarray,
        slots_np: np.ndarray,
        num_logprobs_np: np.ndarray,
        width: int,
    ) -> LogprobsTensors | None:
        """Logprobs of the requests emitting this step, picked on the host.

        For steps with a wide request: a canvas of rows for every logprobs
        request, padded to its top-k, would not fit.
        """
        wide_np = num_logprobs_np > self._stash_width
        narrow = None
        if not wide_np.all():
            narrow = self._canvas_rows(
                sampled,
                req_idx_np[~wide_np],
                slots_np[~wide_np],
                num_logprobs_np[~wide_np] > 0,
                width,
            )
        with gpu_sync_allowed():
            emitted_np = num_sampled.cpu().numpy()
        parts: list[LogprobsTensors] = []
        rows_np = np.zeros(num_reqs, dtype=np.int64)
        j = 0
        for req_idx, slot, wide in zip(
            req_idx_np.tolist(), slots_np.tolist(), wide_np.tolist()
        ):
            k = int(emitted_np[req_idx])
            part = None
            if wide:
                if k:
                    part = self._wide_logprobs.pop(slot, None)
            else:
                assert narrow is not None
                if k:
                    part = LogprobsTensors(
                        logprob_token_ids=narrow.logprob_token_ids[j, :k],
                        logprobs=narrow.logprobs[j, :k],
                        selected_token_ranks=narrow.selected_token_ranks[j, :k],
                    )
                j += 1
            if part is not None:
                parts.append(part)
                rows_np[req_idx] = part.logprobs.shape[0]
        if not parts:
            return None
        return concat_logprob_stashes(parts, (np.cumsum(rows_np) - rows_np).tolist())

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill,
        # init canvas for any new prefills, stage decode slot indices. ---
        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        # Real canvas length per decode request; == CL except for a canvas
        # truncated near max_model_len.
        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_tensor_h2d(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        if num_decode > 0:
            token_slots = decode_slots.repeat_interleave(
                valid_canvas_len, output_size=int(valid_canvas_len_np.sum())
            )
            top_k, top_p, _ = self.sampling_states.get_top_k_top_p(
                token_slots, decode_slots_np, logits.shape[-1]
            )
            if top_k is not None or top_p is not None:
                logits = _filter_sampling_logits(
                    logits,
                    self.sampling_states.temperature.gpu[token_slots],
                    top_k,
                    top_p,
                    self.mask_token_id,
                )

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Padded positions are zeroed (uniform logits) and are treated
        # as non-masked (never transferred or committed).
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)].masked_fill(~valid.reshape(-1, 1), 0)

        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()

        all_slots = input_batch.idx_mapping[:num_reqs]

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit->False, converge->True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)

        req_temps = self.sampling_states.temperature.gpu[decode_slots]

        logits_3d = _compiled_masked_step(
            logits[: num_decode * CL],
            decode_slots,
            decode_idx,
            all_slots,
            valid_canvas_len,
            req_temps,
            # State
            states.canvas,
            states.step,
            states.is_encoder_phase,
            states.unmask_logprobs,
            states.unmask_ranks,
            # Output
            sampled,
            num_sampled,
            self.req_states.draft_tokens,
            # Config
            max_denoising_steps=int(states.max_denoising_steps),
            mask_token_id=int(states.mask_token_id),
            CL=CL,
            all_greedy=bool(
                (self.sampling_states.temperature.np[decode_slots_np] == 0).all()
            ),
            capture_logprobs=max_num_logprobs >= 0,
            leftmost=self.leftmost,
            threshold_mode=self.threshold_mode,
            log_threshold=self.log_threshold,
        )

        logprobs_tensors = None
        if max_num_logprobs >= 0 and num_decode > 0:
            dec_num_logprobs_np = self.sampling_states.num_logprobs[decode_slots_np]
            wide_np = dec_num_logprobs_np > self._stash_width
            narrow_np = (dec_num_logprobs_np >= 0) & ~wide_np
            topk_rows_np = np.flatnonzero(narrow_np & (dec_num_logprobs_np > 0))
            wide_rows_np = np.flatnonzero(wide_np)
            # Every narrow row gets the batch's widest narrow top-k, as
            # compute_topk_scores gives it.
            num_logprobs = int(dec_num_logprobs_np[narrow_np].max(initial=0))
            just_converged = states.is_encoder_phase[decode_slots] & ~is_committing
            if len(topk_rows_np) > 0:
                # Which slots converged is only known on the device, so the
                # stash write is masked instead of picked on the host.
                topk_logits, topk_slots, keep = logits_3d, decode_slots, just_converged
                if len(topk_rows_np) < num_decode:
                    topk_rows = async_tensor_h2d(
                        topk_rows_np.astype(np.int64), device=device
                    )
                    topk_logits = logits_3d[topk_rows]
                    topk_slots = decode_slots[topk_rows]
                    keep = just_converged[topk_rows]
                self._stash_topk_logprobs(
                    topk_logits,
                    states.canvas[topk_slots],
                    topk_slots,
                    keep,
                    num_logprobs,
                )
            lp_rows_np = np.flatnonzero(dec_num_logprobs_np >= 0)
            if len(wide_rows_np) > 0:
                self._pend_wide_logprobs(
                    logits_3d,
                    states.canvas[decode_slots],
                    wide_rows_np,
                    decode_slots_np,
                    valid_canvas_len_np,
                    just_converged,
                    int(dec_num_logprobs_np.max()),
                )
                logprobs_tensors = self._emitted_logprobs(
                    num_reqs,
                    sampled,
                    num_sampled,
                    decode_indices_np[lp_rows_np],
                    decode_slots_np[lp_rows_np],
                    dec_num_logprobs_np[lp_rows_np],
                    1 + num_logprobs,
                )
            elif len(lp_rows_np) > 0:
                logprobs_tensors = self._canvas_logprobs(
                    num_reqs,
                    sampled,
                    decode_indices_np[lp_rows_np],
                    decode_slots_np[lp_rows_np],
                    dec_num_logprobs_np[lp_rows_np] > 0,
                    1 + num_logprobs,
                )

        return self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            logprobs_tensors=logprobs_tensors,
        )
