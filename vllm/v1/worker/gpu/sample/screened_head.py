# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k logits from an FP8 screen of the lm_head plus exact logits.

An FP8 copy of the lm_head scores every row j as s_j. Row j's logit as the full
head computes it before rounding, c_j, satisfies |c_j - s_j| <= ||x|| * err_j:
Cauchy-Schwarz on W_j - W8_j plus the fp32 accumulation error of both GEMMs.
Let t be the k-th largest per-block maximum of s_j - ||x|| err_j: at least k
rows have c_j >= t. A row whose upper bound is below t - 2 eps |t| sits more
than one output ulp below each of them, so it can neither enter nor tie the
full head's top k. The remaining rows get exact logits and every other row
-inf. Greedy sampling (k = 1) and top-k sampling only read the top k, so they
return the full head's token for the same seed (up to sub-ulp ties that depend
on summation order, as with any GEMM change). The screen reads half the bytes
of a BF16 head.
"""

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.sampler import Sampler

_QUANT_CHUNK = 4096


@triton.jit
def _screen_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    err_ptr,
    ub_ptr,
    block_max_ptr,
    num_tokens,
    vocab_size,
    hidden_size,
    BLOCK_N: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * BLOCK_V + tl.arange(0, BLOCK_V)
    toks = tl.arange(0, BLOCK_N)
    row_mask = rows < vocab_size
    tok_mask = toks < num_tokens
    acc = tl.zeros([BLOCK_N, BLOCK_V], dtype=tl.float32)
    sq = tl.zeros([BLOCK_N], dtype=tl.float32)
    for h0 in range(0, hidden_size, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        col_mask = cols < hidden_size
        x = tl.load(
            x_ptr + toks[:, None] * hidden_size + cols[None, :],
            mask=tok_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + rows[None, :].to(tl.int64) * hidden_size + cols[:, None],
            mask=row_mask[None, :] & col_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(x, w.to(x.dtype), acc)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, axis=1)
    s = acc * tl.load(scale_ptr + rows, mask=row_mask, other=0.0)[None, :]
    bound = tl.sqrt(sq)[:, None] * tl.load(err_ptr + rows, mask=row_mask, other=0.0)
    mask = tok_mask[:, None] & row_mask[None, :]
    lb = tl.where(mask, s - bound, float("-inf"))
    tl.store(
        block_max_ptr + toks * tl.num_programs(0) + block,
        tl.max(lb, axis=1),
        mask=tok_mask,
    )
    tl.store(
        ub_ptr + toks[:, None].to(tl.int64) * vocab_size + rows[None, :],
        s + bound,
        mask=mask,
    )


@triton.jit
def _select_kernel(
    ub_ptr,
    t_ptr,
    logits_ptr,
    cand_ptr,
    num_cand_ptr,
    vocab_size,
    TIE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_size
    t = tl.load(t_ptr + tok)
    ub = tl.load(ub_ptr + tok * vocab_size + cols, mask=mask, other=float("-inf"))
    keep = mask & (ub >= t - TIE * tl.abs(t))
    tl.store(logits_ptr + tok * vocab_size + cols, float("-inf"), mask=mask)
    num_keep = tl.sum(keep.to(tl.int32), axis=0)
    if num_keep > 0:
        base = tl.atomic_add(num_cand_ptr + tok, num_keep)
        offs = base + tl.cumsum(keep.to(tl.int32), axis=0) - 1
        tl.store(cand_ptr + tok * vocab_size + offs, cols, mask=keep)


@triton.jit
def _exact_kernel(
    x_ptr,
    w_ptr,
    cand_ptr,
    num_cand_ptr,
    logits_ptr,
    vocab_size,
    hidden_size,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    num_cand = tl.load(num_cand_ptr + tok)
    step = tl.num_programs(1) * BLOCK_R
    for start in range(tl.program_id(1) * BLOCK_R, num_cand, step):
        r = start + tl.arange(0, BLOCK_R)
        r_mask = r < num_cand
        rows = tl.load(cand_ptr + tok * vocab_size + r, mask=r_mask, other=0)
        rows = rows.to(tl.int64)
        acc = tl.zeros([BLOCK_R], dtype=tl.float32)
        for h0 in range(0, hidden_size, BLOCK_H):
            cols = h0 + tl.arange(0, BLOCK_H)
            col_mask = cols < hidden_size
            x = tl.load(x_ptr + tok * hidden_size + cols, mask=col_mask, other=0.0)
            w = tl.load(
                w_ptr + rows[:, None] * hidden_size + cols[None, :],
                mask=r_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
        tl.store(
            logits_ptr + tok * vocab_size + rows,
            acc.to(logits_ptr.dtype.element_ty),
            mask=r_mask,
        )


class ScreenedLMHead:
    """lm_head logits for batches that only read the top k, exact on every row
    that can be among them and -inf elsewhere."""

    # Tiles and caps were measured on GB10 (lm_head 248320 x 2560): greedy, the
    # screen beats the BF16 head by 39% at 64 tokens and loses beyond 96, where
    # the GEMM is no longer bandwidth bound; top-32 still wins by 23% at 64.
    MAX_TOKENS = 64
    MAX_TOP_K = 32
    BLOCK_V = 64
    BLOCK_H = 256
    SELECT_BLOCK = 4096
    EXACT_PROGRAMS = 64

    def __init__(
        self,
        weight: torch.Tensor,
        vocab_size: int,
        sampler: Sampler,
        max_num_tokens: int,
    ):
        self.weight = weight[:vocab_size]
        self.sampler = sampler
        self.vocab_size, self.hidden_size = self.weight.shape
        self.max_num_tokens = min(max_num_tokens, self.MAX_TOKENS)
        # One output ulp is at most eps * |logit|; argmax ties resolve to the
        # lowest index.
        self.tie = 2 * torch.finfo(weight.dtype).eps
        self.w_fp8 = torch.empty_like(self.weight, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(self.vocab_size, device=weight.device)
        self.err = torch.empty(self.vocab_size, device=weight.device)
        self.ub = torch.empty(
            self.max_num_tokens, self.vocab_size, device=weight.device
        )
        self.cand = torch.empty_like(self.ub, dtype=torch.int32)
        self.block_max = torch.empty(
            self.max_num_tokens,
            triton.cdiv(self.vocab_size, self.BLOCK_V),
            device=weight.device,
        )
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the FP8 copy and its error bounds from the lm_head weight."""
        u = 2.0**-24
        # fp32 accumulation error over hidden_size terms, relative to
        # sum_i |x_i w_i| <= ||x|| ||w||.
        gamma = self.hidden_size * u / (1 - self.hidden_size * u)
        for start in range(0, self.vocab_size, _QUANT_CHUNK):
            rows = slice(start, start + _QUANT_CHUNK)
            _, scale = ops.scaled_fp8_quant(
                self.weight[rows],
                use_per_token_if_dynamic=True,
                output=self.w_fp8[rows],
            )
            self.scale[rows] = scale.view(-1)
            w = self.weight[rows].double()
            deq = self.w_fp8[rows].double() * scale.double()
            err = (
                (w - deq).norm(dim=1)
                + gamma * w.norm(dim=1)
                + (gamma + 2 * u) * deq.norm(dim=1)
            )
            # Slack for the fp32 norm of x and the fp32 bound arithmetic.
            self.err[rows] = err * (1 + 4 * gamma)

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        sampler: Sampler,
        vocab_size: int,
        max_num_tokens: int,
    ) -> "ScreenedLMHead":
        get_language_model = getattr(model, "get_language_model", None)
        language_model = get_language_model() if get_language_model else model
        lm_head = getattr(language_model, "lm_head", None)
        processor = getattr(language_model, "logits_processor", None)
        reason = None
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(89)
        ):
            reason = "an FP8-capable CUDA GPU (compute capability 8.9+)"
        elif envs.VLLM_BATCH_INVARIANT:
            # Whether a request is screened depends on the rest of its batch.
            reason = "VLLM_BATCH_INVARIANT to be off"
        elif (
            type(sampler) is not Sampler
            or sampler.compute_nans
            or sampler.trace_replay_state is not None
            or sampler.return_sampling_mask
        ):
            reason = "the default sampler without NaN checks or trace replay"
        elif not isinstance(lm_head, VocabParallelEmbedding) or not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            reason = "an unquantized lm_head"
        elif lm_head.tp_size > 1 or getattr(lm_head, "bias", None) is not None:
            reason = "an lm_head without TP sharding or bias"
        elif lm_head.weight.dtype not in (torch.bfloat16, torch.float16):
            reason = "a BF16 or FP16 lm_head"
        elif (
            not isinstance(processor, LogitsProcessor)
            or processor.logits_as_input
            or processor.soft_cap is not None
            or processor.scale != 1.0
            or processor.head_dtype not in (None, lm_head.weight.dtype)
        ):
            reason = "logits that are the plain lm_head projection (no LoRA)"
        if reason is not None:
            raise ValueError(f"screened_lm_head requires {reason}.")
        assert lm_head is not None
        weight = lm_head.weight.data
        head = cls(weight, vocab_size, sampler, max_num_tokens)
        # Compile the kernels now rather than on the first request.
        for num_tokens in (1, 17, 33):
            if num_tokens <= head.max_num_tokens:
                head(torch.randn_like(weight[:num_tokens]))
        return head

    def required_top_k(self, idx_mapping_np: np.ndarray, num_tokens: int) -> int | None:
        """How many top logits sampling the batch reads (1 when all greedy), or
        None when it needs more than the screen certifies."""
        sampler = self.sampler
        states = sampler.sampling_states
        if (
            not 0 < num_tokens <= self.max_num_tokens
            or np.any(sampler.uses_logits_processors[idx_mapping_np])
            or sampler.get_logprobs_dims(idx_mapping_np) is not None
        ):
            return None
        sampled = idx_mapping_np[states.temperature.np[idx_mapping_np] != 0.0]
        if sampled.size == 0:
            return 1
        # Without top-k, top-p reads the whole distribution; min-p is applied
        # before top-k.
        top_k = int(states.top_k.np[sampled].max())
        if top_k > min(self.MAX_TOP_K, self.block_max.shape[1]) or np.any(
            states.min_p.np[sampled] != 0.0
        ):
            return None
        return top_k

    def __call__(self, x: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        num_tokens = x.shape[0]
        vocab_size = self.vocab_size
        assert num_tokens <= self.max_num_tokens
        x = x.contiguous()
        ub = self.ub[:num_tokens]
        cand = self.cand[:num_tokens]
        block_max = self.block_max[:num_tokens]
        # A single token block, so the FP8 weights are read once.
        block_n = max(16, triton.next_power_of_2(num_tokens))
        _screen_kernel[(triton.cdiv(vocab_size, self.BLOCK_V),)](
            x,
            self.w_fp8,
            self.scale,
            self.err,
            ub,
            block_max,
            num_tokens,
            vocab_size,
            self.hidden_size,
            BLOCK_N=block_n,
            BLOCK_V=self.BLOCK_V,
            BLOCK_H=self.BLOCK_H,
        )
        t = block_max.topk(top_k, dim=-1).values[:, -1].contiguous()
        logits = torch.empty(num_tokens, vocab_size, device=x.device, dtype=x.dtype)
        num_cand = torch.zeros(num_tokens, device=x.device, dtype=torch.int32)
        _select_kernel[(num_tokens, triton.cdiv(vocab_size, self.SELECT_BLOCK))](
            ub,
            t,
            logits,
            cand,
            num_cand,
            vocab_size,
            TIE=self.tie,
            BLOCK=self.SELECT_BLOCK,
        )
        _exact_kernel[(num_tokens, self.EXACT_PROGRAMS)](
            x,
            self.weight,
            cand,
            num_cand,
            logits,
            vocab_size,
            self.hidden_size,
            BLOCK_R=4,
            BLOCK_H=self.BLOCK_H,
        )
        return logits
