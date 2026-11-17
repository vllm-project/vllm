# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov draft sampling, one launch per draft step (ROCm).

``DSparkSpeculator._sample_sequential`` runs, for each of the ``k`` draft steps
(k = 7 for Kimi-K3.1), the Markov embedding of the previous token, the bias
GEMV over the whole vocabulary (``markov_w2``: 163840 x 512 bf16, 168 MB,
replicated), the bf16 add to the base logits, the Gumbel sample (its kernel, a
torch argmax over the block maxima and a gather) and the copy into
``draft_tokens``: about nine launches a step, ~45 us a step on MI355X at c1.

Here a step is one launch over the vocabulary in blocks of ``BLOCK`` tokens.
A program (request, block) takes the previous token (step 0: the anchor token;
later steps: the largest of the previous step's packed keys, and program 0
writes it into ``draft_tokens``),
computes the block's bias from ``markov_w1[prev]`` (FP32 accumulation, rounded
to bf16 as the GEMV output), adds the base logits with one bf16 rounding (a
bf16 tensor add), stores the logits into the draft logits cache (honouring
``DraftLogitsCache.row_mapping``) and keeps the block's Gumbel-max with vLLM's
own ``gumbel_noised_argmax`` (same seed mixing with the draft step, same
drafting salt and position, same temperature division). The block's maximum is
packed into one 64-bit key (orderable FP32 value high, inverted token id low,
so the largest key is the largest value and, among equal values, the first
token, as vLLM's two-level argmax) and merged with an atomic max into one of
``NSUB`` sub-slots of the step (spreading the atomics). A last small launch
selects the last step's token and clears the slots for the next call.

Greedy drafts (no draft logits cache) keep the block's plain argmax instead
(``torch.argmax``'s first maximum), and cache nothing.

Exactness: the sampling is vLLM's. Only the summation order of the 512-term
bias dot product differs from the GEMV's (as with any GEMM tiling), so a bias
value can differ in its last bf16 bit. Graph capturable: fixed launches, slots
allocated (zeroed) once and cleared by the launches themselves.

Enabled by VLLM_K31_BEAT_MARKOV (default 1) under
decode_backend=latency_specialized only, for the Kimi-K3.1 DSpark drafts
(``MARKOV_DRAFTS``): the preview's MLA draft (K3DSparkForCausalLM) and the
release's GQA sliding-window draft (Qwen3DSparkForCausalLM, from the speculators
checkpoint). Both run the same DSparkMarkovHead over the target vocabulary.
tests/kernels/models/amd/test_k31_fused_markov.py compares it with vLLM's loop.
deepseek_v41/amd/ops/fused_markov.py is the same scheme for DeepSeek-V4.1 on
gfx942, reached through the draft model's ``sample_sequential``. That hook
would turn off the batch-sharded draft logits cache
(``supports_batch_sharded_draft_logits``), which these launches honour
(``row_mapping``), so the speculator's ``_sample_sequential`` is patched to call
them instead (``specialized_decode/levers/dspark.py``).
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed
from vllm.v1.worker.gpu.spec_decode.draft_logits import draft_logits_row

logger = init_logger(__name__)

# Vocabulary tokens a program (163840 / 32 = 5120 programs a request), the
# Markov rank columns multiplied at a time and the program's warps.
BLOCK = 32
K_CHUNK = 128
NUM_WARPS = 4
# Sub-slots a step's atomic maxima spread over (same-address atomics serialize).
NSUB = 64
# Requests beyond which the stock loop runs: a request is a grid row that reads
# markov_w2 again, so from 2 requests on (c1 decode has 1 a rank) the stock
# loop's one GEMM a step is as fast or faster.
MAX_ROWS = 1


@triton.jit
def _pack_key(val, tok):
    # Orderable FP32 (-0 folded into +0) in the high word, the inverted token
    # id in the low word: the largest key is the largest value and, among
    # equal values, the first token.
    val = tl.where(val == 0.0, 0.0, val)
    b = val.to(tl.uint32, bitcast=True)
    o = tl.where((b >> 31) != 0, ~b, b | 0x80000000).to(tl.uint64)
    return (o << 32) | (0xFFFFFFFF - tok.to(tl.uint32)).to(tl.uint64)


@triton.jit
def _key_token(slots_ptr, NSUB: tl.constexpr):
    key = tl.max(tl.load(slots_ptr + tl.arange(0, NSUB)), 0)
    return (0xFFFFFFFF - (key & 0xFFFFFFFF)).to(tl.int64)


@triton.jit
def _k31_markov_step_kernel(
    slots_ptr,  # [>= rows, NSLOT, NSUB] uint64 packed maxima, zero when unused
    input_ids_ptr,
    anchor_idx_ptr,  # [rows] step 0's previous token: input_ids[anchor_idx[r]]
    tokens_ptr,  # draft_tokens [max_num_reqs, k]
    tokens_stride,
    w1_ptr,  # markov_w1 [V1, RANK] bf16
    w2_ptr,  # markov_w2 [>= vocab, RANK] bf16
    base_ptr,  # base logits of row r for this step at r * base_stride + v
    base_stride,
    cache_ptr,  # draft logits cache [rows, k, V]
    cache_stride0,
    cache_stride1,
    row_map_ptr,  # DraftLogitsCache.row_mapping or None
    idx_map_ptr,  # [rows, k] request state index
    idx_stride,
    pos_ptr,  # [rows, k] sampled token position
    pos_stride,
    temp_ptr,
    seeds_ptr,
    vocab,
    GREEDY: tl.constexpr,
    STEP: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    K_CHUNK: tl.constexpr,
    NSLOT: tl.constexpr,
    NSUB: tl.constexpr,
):
    r64 = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1)
    v = p * BLOCK + tl.arange(0, BLOCK)
    vmask = v < vocab
    rslots = slots_ptr + r64 * NSLOT * NSUB
    if STEP == 0:
        prev = tl.load(input_ids_ptr + tl.load(anchor_idx_ptr + r64)).to(tl.int64)
    else:
        prev = _key_token(rslots + (STEP - 1) * NSUB, NSUB)
        if p == 0:
            tl.store(tokens_ptr + r64 * tokens_stride + (STEP - 1), prev)
            if STEP >= 2:
                # Nobody reads step STEP - 2's slots any more.
                jj = tl.arange(0, NSUB)
                tl.store(rslots + (STEP - 2) * NSUB + jj, tl.zeros([NSUB], tl.uint64))
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in tl.static_range(0, RANK, K_CHUNK):
        k = k0 + tl.arange(0, K_CHUNK)
        w = tl.load(
            w2_ptr + v[:, None].to(tl.int64) * RANK + k[None, :],
            mask=vmask[:, None],
            other=0.0,
        )
        e = tl.load(w1_ptr + prev * RANK + k).to(tl.float32)
        acc += tl.sum(w.to(tl.float32) * e[None, :], axis=1)
    # The GEMV output is bf16; a bf16 tensor add is one FP32 add rounded once.
    bias = acc.to(tl.bfloat16).to(tl.float32)
    base = tl.load(base_ptr + r64 * base_stride + v, mask=vmask, other=0.0)
    logits = (base.to(tl.float32) + bias).to(tl.bfloat16)
    lf = tl.where(vmask, logits.to(tl.float32), float("-inf"))

    if GREEDY:
        # The speculator's logits.argmax(-1): no cache, no noise.
        value, idx = tl.max(lf, axis=0, return_indices=True)
    else:
        req = tl.load(idx_map_ptr + r64 * idx_stride + STEP).to(tl.int64)
        valid = req >= 0
        row = draft_logits_row(req, row_map_ptr)
        # The cache holds the logits before temperature, as gumbel_sample stores.
        tl.store(
            cache_ptr + row * cache_stride0 + STEP * cache_stride1 + v,
            logits,
            mask=vmask & (row >= 0),
        )
        temp = tl.load(temp_ptr + req, mask=valid, other=0.0).to(tl.float32)
        # gumbel_sample mixes the cache column (the draft step) into the seed
        # and keys a draw by the position before the sampled token.
        seed = spec_step_seed(tl.load(seeds_ptr + req, mask=valid, other=0), STEP)
        pos = tl.load(pos_ptr + r64 * pos_stride + STEP) - 1
        value, idx = gumbel_noised_argmax(
            lf,
            v,
            vmask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=False,
            APPLY_TEMPERATURE=True,
        )
    tl.atomic_max(
        rslots + STEP * NSUB + p % NSUB,
        _pack_key(value.to(tl.float32), p * BLOCK + idx),
        sem="relaxed",
    )


@triton.jit
def _k31_markov_last_kernel(
    slots_ptr,
    tokens_ptr,
    tokens_stride,
    STEP: tl.constexpr,
    NSLOT: tl.constexpr,
    NSUB: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    rslots = slots_ptr + r * NSLOT * NSUB
    tl.store(
        tokens_ptr + r * tokens_stride + STEP, _key_token(rslots + STEP * NSUB, NSUB)
    )
    # Clear the slots the step launches have not cleared (steps STEP - 1, STEP).
    # Another warp may still be reading step STEP's slots: wait for the reads.
    tl.debug_barrier()
    j = tl.arange(0, NSLOT * NSUB)
    tl.store(
        rslots + j, tl.zeros([NSLOT * NSUB], tl.uint64), mask=j >= (STEP - 1) * NSUB
    )


_SLOTS: dict = {}
_OLD_SLOTS: list = []


def _slots(device, nslot, rows):
    """Packed per-step maxima, zeroed once (the launches leave them zeroed) so
    captured graphs keep their address."""
    key = (device, nslot, NSUB)
    s = _SLOTS.get(key)
    if s is None or s.shape[0] < rows:
        if s is not None:
            # Graphs captured with the old slots still use them.
            _OLD_SLOTS.append(s)
        rows = max(rows, MAX_ROWS)
        s = torch.zeros(rows, nslot, NSUB, dtype=torch.uint64, device=device)
        _SLOTS[key] = s
    return s


def markov_drafts() -> tuple[type, ...]:
    """The draft models whose Markov step the launches reproduce: a
    ``DSparkMarkovHead`` (``markov_embed`` = ``markov_w1[prev]``, ``markov_bias``
    = the logits processor's GEMV over the replicated ``markov_w2``), checked at
    Kimi-K3.1's shapes: the preview's MLA draft and the release's GQA
    sliding-window draft (speculators format, served as Qwen3DSpark)."""
    from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    return (K3DSparkForCausalLM, Qwen3DSparkForCausalLM)


def why_not(spec, num_reqs: int) -> str | None:
    """Why this draft step takes vLLM's loop, or None when the fused launches
    compute the same sampling. Checked before the base logits are computed. The
    confidence head is not a reason: ``levers.dspark`` computes it from the
    drafted tokens' Markov embeddings, as vLLM's loop does."""
    model = spec.model
    # Other DSpark drafts keep their loop.
    if not isinstance(model, markov_drafts()):
        return f"not a Kimi-K3.1 DSpark draft ({type(model).__name__})"
    head = getattr(getattr(model, "model", None), "markov_head", None)
    if head is None:
        return "no Markov head"
    # The launches write draft ids as target ids (no map_draft_to_target).
    if getattr(model, "draft_id_to_target_id", None) is not None:
        return "a reduced draft vocabulary"
    if spec.draft_watermarker is not None:
        return "a draft watermarker"
    if spec._d2t_scatter_index is not None:
        return "a reduced draft vocabulary"
    if getattr(spec, "use_fp64_gumbel", False):
        return "FP64 Gumbel noise"
    if spec.acceptance_estimator is not None:
        return "the acceptance estimator"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    lp = getattr(model, "logits_processor", None)
    if lp is not None:
        if lp.scale != 1.0 or lp.soft_cap is not None:
            return "scaled logits"
        if getattr(lp, "head_dtype", None) not in (None, torch.bfloat16):
            return f"a {lp.head_dtype} head dtype"
    if not 1 <= num_reqs <= MAX_ROWS:
        # One reason for every batch size (the caller logs each reason once).
        return f"more than {MAX_ROWS} requests"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return "Markov weights not bf16"
    rank = w1.shape[1]
    if w2.shape[1] != rank or rank % K_CHUNK:
        return f"Markov rank {rank}"
    if not (w1.is_contiguous() and w2.is_contiguous()):
        return "Markov weight layout"
    qm = getattr(head.markov_w2, "quant_method", None)
    if qm is not None and qm.__class__.__name__ != "UnquantizedEmbeddingMethod":
        return "a quantized Markov head"
    if spec.draft_logits is not None and spec.draft_logits.values.dtype != (
        torch.bfloat16
    ):
        return f"a {spec.draft_logits.values.dtype} draft logits cache"
    return None


def run(spec, num_reqs: int, base_logits: torch.Tensor) -> bool:
    """The Markov loop of ``_sample_sequential`` from its base logits
    ([num_reqs, k, V]) on. Returns False, having done nothing, when the base
    logits are not what the fused launch takes."""
    n_spec = spec.num_speculative_steps
    vocab = base_logits.shape[-1]
    head = spec.model.model.markov_head
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    # Greedy drafts: no cache (the launch neither reads nor writes it).
    greedy = spec.draft_logits is None
    cache = base_logits if greedy else spec.draft_logits.values
    row_mapping = None if greedy else spec.draft_logits.row_mapping
    # A step's row is addressed as base_logits[:, step] (stride(0) apart), so
    # only the vocabulary has to be contiguous.
    why = None
    if base_logits.dtype != torch.bfloat16:
        why = f"{base_logits.dtype} base logits"
    elif base_logits.stride(-1) != 1:
        why = "non-contiguous base logits"
    elif w2.shape[0] < vocab:
        why = f"{w2.shape[0]} Markov rows for vocabulary {vocab}"
    elif cache.shape[-1] < vocab:
        why = f"a {cache.shape[-1]}-wide draft logits cache for vocabulary {vocab}"
    if why is not None:
        logger.info_once("DSpark fused Markov sampling off: %s", why)
        return False
    rank = w1.shape[1]
    num_sample = num_reqs * n_spec
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, n_spec)
    assert idx_map.stride(1) == 1 and sample_pos.stride(1) == 1
    tokens = spec.draft_tokens
    nslot = triton.next_power_of_2(n_spec + 1)
    slots = _slots(base_logits.device, nslot, num_reqs)
    nb = triton.cdiv(vocab, BLOCK)
    logger.info_once(
        "DSpark fused Markov sampling (ROCm): %s, %s drafts, %d steps, "
        "vocabulary %d, rank %d",
        type(spec.model).__name__, "greedy" if greedy else "probabilistic",
        n_spec, vocab, rank,
    )  # fmt: skip
    for step in range(n_spec):
        _k31_markov_step_kernel[(num_reqs, nb)](
            slots,
            spec.input_buffers.input_ids,
            spec._anchor_idx,
            tokens,
            tokens.stride(0),
            w1,
            w2,
            base_logits[:, step],
            base_logits.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            row_mapping,
            idx_map,
            idx_map.stride(0),
            sample_pos,
            sample_pos.stride(0),
            spec.temperature,
            spec.seeds,
            vocab,
            GREEDY=greedy,
            STEP=step,
            RANK=rank,
            BLOCK=BLOCK,
            K_CHUNK=K_CHUNK,
            NSLOT=nslot,
            NSUB=NSUB,
            num_warps=NUM_WARPS,
        )
    _k31_markov_last_kernel[(num_reqs,)](
        slots,
        tokens,
        tokens.stride(0),
        STEP=n_spec - 1,
        NSLOT=nslot,
        NSUB=NSUB,
        num_warps=1,
    )
    return True
