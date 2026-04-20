"""Fused Triton kernels for ColBERT/ColPali MaxSim scoring."""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    return 1 << (x - 1).bit_length()

# ---------------------------------------------------------------------------
# Hardware-aware autotune configs
# ---------------------------------------------------------------------------

def _detect_gpu():
    """Detect GPU architecture family via compute capability.

    Uses torch.cuda.get_device_capability() (not string matching) following
    FlashAttention's approach. Maps to architecture families:
      sm_80       = Ampere datacenter (A100, A800)
      sm_86       = Ampere consumer  (RTX 3090, A10, A40, A6000)
      sm_89       = Ada Lovelace     (RTX 4090, L40S, L4)
      sm_90/90a   = Hopper           (H100, H200)
      sm_100/100a = Blackwell DC     (B100, B200, GB200)
      sm_120      = Blackwell consumer (RTX 5090, DGX Spark)
    """
    if not torch.cuda.is_available():
        return "generic"
    major, minor = torch.cuda.get_device_capability()
    if major == 9:
        return "hopper"       # sm_90: H100, H200
    if major >= 10:
        return "blackwell"    # sm_100+: B200, RTX 5090, etc.
    if major == 8:
        if minor == 0:
            return "a100"     # sm_80: A100
        if minor >= 9:
            return "ada"      # sm_89: RTX 4090, L40S
        return "ampere"       # sm_86: RTX 3090, A10, A40
    return "generic"          # V100 (sm_70), T4 (sm_75), etc.


def _get_configs(gpu=None):
    gpu = gpu or _detect_gpu()
    # Small blocks for large embedding dims (d=512, 1024, 2048)
    large_d_configs = [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 16}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 32}, num_warps=4, num_stages=2),
    ]
    if gpu == "hopper":
        # H100: 228 KB SMEM, WGMMA via tl.dot, TMA automatic, num_stages=3-4
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        ]
    if gpu == "blackwell":
        # B200: 228 KB SMEM + 256 KB TMEM, deeper pipelines possible
        # Use aggressive configs; Triton 3.6+ has early Blackwell support
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        ]
    if gpu == "a100":
        # A100 sm_80: 164 KB SMEM, num_stages=1-2 typical
        # stages=2 variants: double-buffer D loads, overlap with compute.
        # SMEM for stages=2 at (128,64): 2*(128*128+64*128)*2 + 128*64*4 = 128 KB, fits.
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        ]
    if gpu == "ada":
        # RTX 4090 / L40S sm_89: similar to A100 but 8 warps for larger blocks
        # Half the memory BW of A100, same FP16 tensor core perf
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 128, "BLOCK_D": 64}, num_warps=8, num_stages=1),
        ]
    if gpu == "ampere":
        # RTX 3090 / A10 / A40 sm_86: less SMEM than A100
        return large_d_configs + [
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=1),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=1),
        ]
    # generic: V100, T4, etc. — conservative configs
    return large_d_configs + [
        triton.Config({"BLOCK_Q": 16, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_D": 128}, num_warps=8, num_stages=2),
    ]


def _prune_configs(configs, named_args, **kwargs):
    Lq = named_args.get("Lq", 32)
    d = named_args.get("d", 128)
    # SMEM limits: A100=164KB, H100=228KB. Use 200KB as safe default;
    # Triton autotune will skip configs that exceed actual hardware SMEM.
    smem_limit = 200_000
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
    else:
        major = 0
    if major >= 9:  # Hopper+
        smem_limit = 220_000
    pruned = []
    for cfg in configs:
        bq, bd = cfg.kwargs["BLOCK_Q"], cfg.kwargs["BLOCK_D"]
        if bq > Lq * 2:
            continue
        if (bq * d + bd * d) * 2 + bq * bd * 4 > smem_limit:
            continue
        pruned.append(cfg)
    return pruned or configs[:4]


_CONFIGS = _get_configs()

# ---------------------------------------------------------------------------
# Fixed-config kernel for small inputs (no autotune overhead)
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_fwd_kernel_small(
    Q_ptr, D_ptr, lengths_ptr, scores_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    shared_docs: tl.constexpr,
    BLOCK_Q: tl.constexpr = 32, BLOCK_D: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    if shared_docs:
        doc_idx = pid // Nq
        q_idx = pid % Nq
    else:
        q_idx = pid // B
        doc_idx = pid % B
    if q_idx >= Nq:
        return

    # Cast to int64 to avoid int32 overflow in pointer arithmetic at large N
    # (doc_idx * stride_d_b can exceed INT32_MAX when N*Ld*d > 2^31)
    d_batch = tl.cast(doc_idx if shared_docs else q_idx * B + doc_idx, tl.int64)
    doc_len = tl.load(lengths_ptr + d_batch).to(tl.int32)

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < Lq

        Q_ptrs = (Q_ptr + q_idx * stride_q_n
                  + q_off[:, None] * stride_q_l
                  + k_off[None, :] * stride_q_d)
        Q_block = tl.load(
            Q_ptrs,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_ptrs = (D_ptr + d_batch * stride_d_b
                      + d_off[:, None] * stride_d_l
                      + k_off[None, :] * stride_d_d)
            D_block = tl.load(
                D_ptrs,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))
            m = tl.maximum(m, tl.max(S, axis=1))

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

    tl.store(scores_ptr + q_idx * stride_s_n + doc_idx * stride_s_b, score_acc)


# Threshold: below this B*Lq*Ld, use fixed-config kernel (no autotune)
_SMALL_THRESHOLD = 500_000


# ---------------------------------------------------------------------------
# Unified forward kernel (single-query & batched)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_CONFIGS, key=["Lq", "d_pad"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def _maxsim_fwd_kernel(
    Q_ptr, D_ptr, lengths_ptr, q_lengths_ptr, scores_ptr, argmax_ptr,
    Nq, B,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    shared_docs: tl.constexpr,
    save_argmax: tl.constexpr,
    use_q_lengths: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grid mapping: when shared_docs, use doc-major order so consecutive CTAs
    # access the SAME D doc across Q chunks → D stays in L1/L2, loaded ~1x
    # instead of Nq times. For ColPali (Nq=8 Q chunks), this reduces D HBM
    # reads from 8x to ~1x.
    if shared_docs:
        doc_idx = pid // Nq
        q_idx = pid % Nq
    else:
        q_idx = pid // B
        doc_idx = pid % B
    if q_idx >= Nq:
        return

    # Cast to int64 to avoid int32 overflow in pointer arithmetic at large N
    # (doc_idx * stride_d_b can exceed INT32_MAX when N*Ld*d > 2^31)
    d_batch = tl.cast(doc_idx if shared_docs else q_idx * B + doc_idx, tl.int64)
    doc_len = tl.load(lengths_ptr + d_batch).to(tl.int32)
    q_len = tl.load(q_lengths_ptr + q_idx).to(tl.int32) if use_q_lengths else Lq

    k_off = tl.arange(0, d_pad)
    k_mask = k_off < d
    score_acc = tl.zeros([], dtype=tl.float32)

    for q_start in tl.static_range(0, Lq, BLOCK_Q):
        q_off = q_start + tl.arange(0, BLOCK_Q)
        q_valid = q_off < q_len

        Q_ptrs = (Q_ptr + q_idx * stride_q_n
                  + q_off[:, None] * stride_q_l
                  + k_off[None, :] * stride_q_d)
        Q_block = tl.load(
            Q_ptrs,
            mask=q_valid[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float16)

        m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        m_idx = tl.full([BLOCK_Q], 0, dtype=tl.int32)

        for d_start in range(0, Ld, BLOCK_D):
            d_off = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_off < doc_len

            D_ptrs = (D_ptr + d_batch * stride_d_b
                      + d_off[:, None] * stride_d_l
                      + k_off[None, :] * stride_d_d)
            D_block = tl.load(
                D_ptrs,
                mask=d_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            S = tl.dot(Q_block, tl.trans(D_block))
            S = tl.where(d_valid[None, :], S, float("-inf"))

            tile_max = tl.max(S, axis=1)
            if save_argmax:
                tile_argmax = tl.argmax(S, axis=1).to(tl.int32) + d_start
                update = tile_max > m
                m_idx = tl.where(update, tile_argmax, m_idx)
            m = tl.maximum(m, tile_max)

        m = tl.where(q_valid, m, 0.0)
        score_acc += tl.sum(m)

        if save_argmax:
            tl.store(
                argmax_ptr + pid * Lq + q_off, m_idx, mask=q_valid,
            )

    tl.store(scores_ptr + q_idx * stride_s_n + doc_idx * stride_s_b, score_acc)


# ---------------------------------------------------------------------------
# Backward kernels for training
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_bwd_dQ_kernel(
    D_ptr, argmax_ptr, grad_s_ptr, grad_Q_ptr,
    B: tl.constexpr, Lq: tl.constexpr, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
):
    q_idx = tl.program_id(0)
    k = tl.arange(0, d_pad)
    km = k < d
    acc = tl.zeros([d_pad], dtype=tl.float32)
    for b in range(B):
        gs = tl.load(grad_s_ptr + b).to(tl.float32)
        j = tl.load(argmax_ptr + b * Lq + q_idx)
        v_ptrs = D_ptr + b * stride_d_b + j * stride_d_l + k * stride_d_d
        v = tl.load(v_ptrs, mask=km, other=0.0).to(tl.float32)
        acc += gs * v
    tl.store(grad_Q_ptr + q_idx * d + k, acc.to(tl.float16), mask=km)


@triton.jit
def _maxsim_bwd_dD_kernel(
    Q_ptr, argmax_ptr, grad_s_ptr, grad_D_ptr,
    Lq: tl.constexpr, Ld, d: tl.constexpr, d_pad: tl.constexpr,
    stride_d_b, stride_d_l, stride_d_d,
    stride_q_l, stride_q_d,
):
    doc_id = tl.program_id(0)
    k = tl.arange(0, d_pad)
    km = k < d
    gs = tl.load(grad_s_ptr + doc_id).to(tl.float32)
    for q_idx in range(Lq):
        j = tl.load(argmax_ptr + doc_id * Lq + q_idx)
        qv_ptrs = Q_ptr + q_idx * stride_q_l + k * stride_q_d
        qv = tl.load(qv_ptrs, mask=km, other=0.0).to(tl.float32)
        tl.atomic_add(
            grad_D_ptr + doc_id * stride_d_b + j * stride_d_l + k * stride_d_d,
            (gs * qv).to(tl.float16), mask=km,
        )


# ---------------------------------------------------------------------------
# Helper: launch the unified kernel
# ---------------------------------------------------------------------------

def _launch_fwd(
    Q, D, lengths, Nq, B, Lq, Ld, d, shared_docs, save_argmax, q_lengths=None,
):
    d_pad = _next_pow2(d)
    scores = torch.empty(Nq, B, device=Q.device, dtype=torch.float32)
    if save_argmax:
        argmax = torch.empty(
            Nq * B, Lq, device=Q.device, dtype=torch.int32,
        )
    else:
        argmax = Q  # dummy
    use_q_lengths = q_lengths is not None
    # dummy q_lengths if not provided (not read by kernel when use_q_lengths=0)
    q_lengths_t = q_lengths if use_q_lengths else lengths

    # For small inputs, use fixed-config kernel (no autotune overhead)
    # Skip for large d — fixed block sizes exceed shared memory
    if not save_argmax and Nq * B < 500 and d <= 512:
        _maxsim_fwd_kernel_small[(Nq * B,)](
            Q, D, lengths, scores,
            Nq, B, Lq, Ld, d, d_pad,
            Q.stride(-3) if Q.dim() == 3 else 0, Q.stride(-2), Q.stride(-1),
            D.stride(0), D.stride(1), D.stride(2),
            scores.stride(0), scores.stride(1),
            1 if shared_docs else 0,
        )
        return scores, argmax

    _maxsim_fwd_kernel[(Nq * B,)](
        Q, D, lengths, q_lengths_t, scores, argmax,
        Nq, B, Lq, Ld, d, d_pad,
        Q.stride(-3) if Q.dim() == 3 else 0, Q.stride(-2), Q.stride(-1),
        D.stride(0), D.stride(1), D.stride(2),
        scores.stride(0), scores.stride(1),
        1 if shared_docs else 0,
        1 if save_argmax else 0,
        1 if use_q_lengths else 0,
    )
    return scores, argmax


def _default_lengths(B, Ld, device, doc_lengths=None):
    if doc_lengths is not None:
        return doc_lengths.to(torch.int32).contiguous()
    return torch.full((B,), Ld, device=device, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_maxsim(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None,
                 query_chunk_size: int | None = 128,
                 splitk: bool = False) -> torch.Tensor:
    """Compute MaxSim scores.

    Supports two D layouts:
      Padded:  D: [B, Ld, d], doc_lengths: [B] or None  → [B]
      Packed:  D: [total_d, d], doc_lengths: [B+1] cu_seqlens  → [B]

    Args:
        query_chunk_size: Splits the query into chunks of this size and
            computes MaxSim per chunk with the batched shared-docs kernel,
            then sums. Benefits:
            - Better GPU occupancy (more thread blocks)
            - Single autotune config (chunk_size, Ld) for any Lq
            - Faster warmup / fewer Triton compilations
            Default 128. Set to None to disable chunking.
            Only used for padded D path.
        splitk: Enable split-K over Ld (default False). Set True for the
            ColPali regime (Lq>=512, Ld>=512) at small B — splits Ld into
            ~128-token chunks per CTA for better L1 locality and SM
            utilisation. num_splits is chosen as min(Ld//128, 16).
            Use flash_maxsim_splitk() directly for manual num_splits control.
            Only used for padded D path.
    """
    assert Q.dim() == 2

    # Packed D path: D is [total_d, d], doc_lengths is cu_seqlens [B+1]
    if D.dim() == 2:
        assert doc_lengths is not None, \
            "Packed D [total_d, d] requires doc_lengths as cu_seqlens [B+1]"
        from .flash_maxsim_varlen import flash_maxsim_packed
        return flash_maxsim_packed(Q, D, doc_lengths)

    assert D.dim() == 3 and Q.shape[1] == D.shape[2]
    Lq, d = Q.shape
    B, Ld, _ = D.shape

    if splitk:
        from .flash_maxsim_advanced import flash_maxsim_splitk
        ns = min(Ld // 128, 16)
        ns = max(ns, 1)
        return flash_maxsim_splitk(Q, D, doc_lengths, num_splits=ns)

    # Chunked path: split Q into fixed-size chunks for better occupancy
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        num_chunks = (Lq + C - 1) // C
        # Pad Q to multiple of C if needed
        if Lq % C != 0:
            Q_padded = torch.nn.functional.pad(Q, (0, 0, 0, num_chunks * C - Lq))
        else:
            Q_padded = Q
        Q_chunked = Q_padded.view(num_chunks, C, d)  # [num_chunks, C, d]
        # Tell kernel the real length of the last chunk
        q_lengths = torch.full((num_chunks,), C, device=Q.device, dtype=torch.int32)
        if Lq % C != 0:
            q_lengths[-1] = Lq % C
        return flash_maxsim_batched(
            Q_chunked, D, doc_lengths=doc_lengths,
            shared_docs=True, query_lengths=q_lengths,
        ).sum(dim=0)  # [num_chunks, B] -> [B]

    # Direct path
    # No Python-level dtype conversion: Triton kernel loads as native dtype
    # and converts to FP16 inline (.to(tl.float16) in the load instruction).
    # This avoids an expensive D-tensor copy for BF16/FP32 inputs.
    Q2 = Q.unsqueeze(0).contiguous()
    D2 = D.contiguous()
    lengths = _default_lengths(B, Ld, D.device, doc_lengths)
    scores, _ = _launch_fwd(Q2, D2, lengths, 1, B, Lq, Ld, d, True, False)
    return scores.squeeze(0)


def _round_up_lq(Lq: int) -> int:
    """Round Lq up to nearest bucket to limit autotune configurations.

    Buckets: 32, 64, 128, 256, 512, 1024, 2048, 4096.
    This gives ~8 compiled variants instead of one-per-Lq, while avoiding
    the 4x Q-loop overhead that rounding 32→128 caused for ColBERT.
    """
    for b in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        if Lq <= b:
            return b
    return _next_pow2(Lq)


def flash_maxsim_batched(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None,
                         shared_docs: bool = True, query_lengths=None,
                         pad_lq: bool = True,
                         query_chunk_size: int | None = 128) -> torch.Tensor:
    """Batched MaxSim. Q: [Nq, Lq, d], D: [B, Ld, d] -> [Nq, B].

    Args:
        query_lengths: optional [Nq] int — variable query lengths per query.
        pad_lq: if True, pad Lq to nearest multiple of 128 to limit autotune
            configurations. query_lengths is set automatically to mask padding.
            Default True.
        query_chunk_size: splits each query into chunks for better GPU
            occupancy. Default 128. Set to None to disable.
    """
    assert Q.dim() == 3 and Q.shape[2] == D.shape[-1]
    Nq, Lq, d = Q.shape

    if shared_docs:
        assert D.dim() == 3
        B, Ld, _ = D.shape
    else:
        assert D.dim() == 4 and D.shape[0] == Nq
        _, B, Ld, _ = D.shape

    # Query chunking: split each [Nq, Lq, d] into [Nq*num_chunks, C, d]
    # then sum chunk scores. Same idea as flash_maxsim's chunked path.
    if query_chunk_size is not None and Lq > query_chunk_size:
        C = query_chunk_size
        num_chunks = (Lq + C - 1) // C
        if Lq % C != 0:
            Q = torch.nn.functional.pad(Q, (0, 0, 0, num_chunks * C - Lq))
        # [Nq, num_chunks, C, d] -> [Nq*num_chunks, C, d]
        Q_chunked = Q.view(Nq, num_chunks, C, d).reshape(Nq * num_chunks, C, d)
        # query_lengths for each chunk
        q_lens_chunk = torch.full(
            (Nq * num_chunks,), C, device=Q.device, dtype=torch.int32,
        )
        if Lq % C != 0:
            # last chunk of each query has fewer real tokens
            last_len = Lq % C
            q_lens_chunk[num_chunks - 1::num_chunks] = last_len
        # Launch: each "query" is now a chunk
        chunk_scores = flash_maxsim_batched(
            Q_chunked, D, doc_lengths=doc_lengths,
            shared_docs=shared_docs, query_lengths=q_lens_chunk,
            pad_lq=pad_lq, query_chunk_size=None,  # no further chunking
        )  # [Nq*num_chunks, B]
        # Sum chunks per original query
        return chunk_scores.view(Nq, num_chunks, B).sum(dim=1)  # [Nq, B]

    # Pad Lq to nearest 128 to limit autotune configs
    if pad_lq and Lq > 0:
        padded_Lq = _round_up_lq(Lq)
        if padded_Lq != Lq:
            if query_lengths is None:
                query_lengths = torch.full(
                    (Nq,), Lq, device=Q.device, dtype=torch.int32,
                )
            Q = torch.nn.functional.pad(Q, (0, 0, 0, padded_Lq - Lq))
            Lq = padded_Lq

    if not shared_docs:
        D = D.reshape(Nq * B, Ld, d)
    Q2 = Q.contiguous()
    D2 = D.contiguous()
    total = B if shared_docs else Nq * B
    lengths = _default_lengths(total, Ld, D.device, doc_lengths)
    q_lens = (
        query_lengths.to(torch.int32).contiguous()
        if query_lengths is not None else None
    )
    scores, _ = _launch_fwd(
        Q2, D2, lengths, Nq, B, Lq, Ld, d, shared_docs, False,
        q_lengths=q_lens,
    )
    return scores


class _FlashMaxSimFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, D):
        Lq, d = Q.shape
        B, Ld, _ = D.shape
        Q2 = Q.unsqueeze(0).contiguous().half()
        D2 = D.contiguous().half()
        lengths = torch.full((B,), Ld, device=D.device, dtype=torch.int32)
        scores, argmax = _launch_fwd(Q2, D2, lengths, 1, B, Lq, Ld, d, True, True)
        ctx.save_for_backward(Q2.squeeze(0), D2, argmax)
        return scores.squeeze(0)

    @staticmethod
    def backward(ctx, grad_scores):
        Q, D, argmax = ctx.saved_tensors
        Lq, d = Q.shape
        B, Ld, _ = D.shape
        grad_scores = grad_scores.contiguous().float()

        grad_Q = torch.zeros_like(Q)
        d_pad = _next_pow2(d)
        _maxsim_bwd_dQ_kernel[(Lq,)](
            D, argmax, grad_scores, grad_Q,
            B, Lq, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
        )
        grad_D = torch.zeros_like(D)
        _maxsim_bwd_dD_kernel[(B,)](
            Q, argmax, grad_scores, grad_D,
            Lq, Ld, d, d_pad,
            D.stride(0), D.stride(1), D.stride(2),
            Q.stride(0), Q.stride(1),
        )
        return grad_Q, grad_D


def flash_maxsim_pairs(q_embs: list, d_embs: list) -> torch.Tensor:
    """Score a list of (query, doc) pairs. Each pair can have different lengths.

    Args:
        q_embs: list of [Lq_i, d] tensors (one per query)
        d_embs: list of [Ld_i, d] tensors (one per doc)

    Returns:
        [N] scores, one per pair
    """
    N = len(q_embs)
    assert len(d_embs) == N and N > 0

    # Always use varlen — fast for both uniform and variable lengths
    from .flash_maxsim_varlen import flash_maxsim_varlen, pack_pairs
    Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld = pack_pairs(q_embs, d_embs)
    return flash_maxsim_varlen(Q_pk, D_pk, cu_q, cu_d, max_lq, max_ld)


def flash_maxsim_train(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """MaxSim with autograd backward. Q: [Lq, d], D: [B, Ld, d] -> [B]."""
    return _FlashMaxSimFn.apply(Q, D)


def maxsim_naive(Q: torch.Tensor, D: torch.Tensor, doc_lengths=None) -> torch.Tensor:
    """Reference PyTorch MaxSim. Q: [Lq, d], D: [B, Ld, d] -> [B]."""
    S = torch.einsum("qd,bld->bql", Q, D)
    if doc_lengths is not None:
        Ld = D.shape[1]
        pos = torch.arange(Ld, device=D.device)[None, None, :]
        S = S.masked_fill(pos >= doc_lengths[:, None, None], float("-inf"))
    return S.max(dim=2).values.sum(dim=1)


# ---------------------------------------------------------------------------
# Persistent (grid-strided) kernel — eliminates per-(query,doc) block launch
# overhead that dominates at small N with varlen.
#
# Instead of launching Nq*B blocks (one per pair), we launch
# num_sms * grid_factor blocks and each block processes multiple pairs via
# a strided loop. At N=10K with avg_Ld=22, this reduces 10K block launches
# to ~216 — recovering the 3-8x speedup hidden by launch overhead.
# ---------------------------------------------------------------------------

@triton.jit
def _maxsim_persistent_fwd(
    Q_ptr, D_ptr, lengths_ptr, scores_ptr,
    total_work, grid_size,
    Nq, B,
    Lq, d: tl.constexpr, d_pad: tl.constexpr,
    stride_q_n, stride_q_l, stride_q_d,
    stride_d_b, stride_d_l, stride_d_d,
    stride_s_n, stride_s_b,
    shared_docs: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    # Constant register arrays — hoisted out of the work loop
    k_off  = tl.arange(0, d_pad)
    k_mask = k_off < d

    # Grid-strided loop: each block claims a strided slice of all work items
    for work_idx in range(pid, total_work, grid_size):
        q_idx   = work_idx // B
        doc_idx = work_idx % B

        # int64 for pointer arithmetic (avoids overflow at large N)
        d_batch  = tl.cast(doc_idx if shared_docs else q_idx * B + doc_idx, tl.int64)
        doc_len  = tl.load(lengths_ptr + d_batch).to(tl.int32)

        score_acc = tl.zeros([], dtype=tl.float32)

        # Q loop — dynamic bound, masked on last block if Lq % BLOCK_Q != 0
        for q_start in range(0, Lq, BLOCK_Q):
            q_off   = q_start + tl.arange(0, BLOCK_Q)
            q_valid = q_off < Lq

            Q_block = tl.load(
                Q_ptr + tl.cast(q_idx, tl.int64) * stride_q_n
                      + q_off[:, None] * stride_q_l
                      + k_off[None, :] * stride_q_d,
                mask=q_valid[:, None] & k_mask[None, :], other=0.0,
            ).to(tl.float16)

            m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

            # D loop — iterates over actual doc_len tokens, not padded max_Ld
            for d_start in range(0, doc_len, BLOCK_D):
                d_off   = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_off < doc_len

                D_block = tl.load(
                    D_ptr + d_batch * stride_d_b
                          + d_off[:, None] * stride_d_l
                          + k_off[None, :] * stride_d_d,
                    mask=d_valid[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float16)

                S = tl.dot(Q_block, tl.trans(D_block))
                S = tl.where(d_valid[None, :], S, float("-inf"))
                m = tl.maximum(m, tl.max(S, axis=1))

            m = tl.where(q_valid, m, 0.0)
            score_acc += tl.sum(m)

        tl.store(
            scores_ptr + tl.cast(q_idx, tl.int64) * stride_s_n
                       + tl.cast(doc_idx, tl.int64) * stride_s_b,
            score_acc,
        )


def flash_maxsim_persistent(
    Q: torch.Tensor,
    D: torch.Tensor,
    doc_lengths=None,
    grid_factor: int = 2,
) -> torch.Tensor:
    """MaxSim with a persistent grid-strided kernel.

    Launches num_sms * grid_factor blocks instead of one block per
    (query, doc) pair. Best for small N (<50K) with variable-length docs
    where standard launch overhead dominates actual compute.

    Args:
        Q:           [Lq, d] or [Nq, Lq, d] float16/bfloat16/float32
        D:           [B, Ld, d]  same dtype
        doc_lengths: [B] int32 actual token counts (None = all Ld)
        grid_factor: blocks per SM. Default 2 (216 blocks on A100).
                     Increase to 4 for better occupancy on short docs.

    Returns: [B] float32 scores  (or [Nq, B] for batched Q)
    """
    batched = Q.dim() == 3
    if batched:
        Nq, Lq, d = Q.shape
    else:
        Lq, d = Q.shape
        Nq = 1

    assert D.dim() == 3 and D.shape[2] == d
    B, Ld, _ = D.shape

    d_pad  = _next_pow2(d)
    scores = torch.empty(Nq, B, device=D.device, dtype=torch.float32)

    Q2 = (Q if batched else Q.unsqueeze(0)).contiguous()
    D2 = D.contiguous()
    lengths = _default_lengths(B, Ld, D.device, doc_lengths)

    num_sms    = torch.cuda.get_device_properties(D.device).multi_processor_count
    grid_size  = num_sms * grid_factor
    total_work = Nq * B

    # Fixed config — no autotune overhead (that's the whole point)
    BLOCK_Q = 32
    BLOCK_D = 64

    _maxsim_persistent_fwd[(grid_size,)](
        Q2, D2, lengths, scores,
        total_work, grid_size,
        Nq, B,
        Lq, d, d_pad,
        Q2.stride(0), Q2.stride(1), Q2.stride(2),
        D2.stride(0), D2.stride(1), D2.stride(2),
        scores.stride(0), scores.stride(1),
        1,          # shared_docs=True
        BLOCK_Q, BLOCK_D,
        num_warps=4, num_stages=2,
    )

    return scores if batched else scores.squeeze(0)
