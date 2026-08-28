# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 FlyDSL Project Contributors
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.mfma_epilogues import (
    c_shuffle_epilog,
    mfma_epilog,
)
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    make_preshuffle_b_layout,
    tile_chunk_coord_i32,
)
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .utils import (
    _if_then,
    flatten_nvfp4_b_tile,
    lds_load_bf16_k64,
    load_bf16_x,
    load_nvfp4_b_tile,
    unflatten_nvfp4_b_tile,
    unpack_b_nvfp4,
)
from .utils import (
    mfma_k64 as _mfma_k64,
)
from .utils import (
    ptr_buffer_resource as _make_ptr_buffer_resource,
)
from .utils import (
    store_x_tile_to_lds as _store_x_tile_to_lds,
)


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    # NOTE: aiter swap passes these for API symmetry; stage1 uses
    # dynamic memrefs so they are ignored.
    doweight_stage1: bool,
    in_dtype: str = "fp8",
    group_size: int = -1,
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    k_batch: int = 1,
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "nvfp4_bf16": W4A16 path: X is bf16, W is packed fp4_e2m1 with fp8_e4m3
        block scales and a global f32 scale (not implemented yet)
    k_batch: Split-K factor. When >1, K is partitioned across k_batch CTAs that
      atomically accumulate gate/up partials. Caller must pre-zero output.
    """

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    if in_dtype != "nvfp4_bf16":
        raise ValueError("only in_dtype='nvfp4_bf16' is supported")
    needs_scale_w = True
    elem_bytes = 2
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")

    # NOTE: don't materialize MLIR types outside an active MLIR Context.
    def out_mlir():
        return (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    # K64-byte micro-step: always 64 bytes per `ku`. For fp16 this is 32 elements.
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )
    if model_dim % tile_k != 0:
        raise ValueError(
            f"stage1 K dimension must be divisible by tile_k, got "
            f"model_dim={model_dim}, tile_k={tile_k}"
        )
    if group_size != 16:
        raise ValueError(
            "FlyDSL nvfp4_bf16 groupwise scale requires group_size=16, "
            f"got {group_size}."
        )
    if inter_dim % tile_n != 0:
        raise ValueError(
            "FlyDSL nvfp4_bf16 stage1 requires inter_dim to be a positive "
            f"multiple of tile_n, got inter_dim={inter_dim}, tile_n={tile_n}."
        )
    use_inloop_w_scale = True
    num_groups = model_dim // 16
    experts * (2 * inter_dim) * num_groups
    # For groupwise scale, weight scale is applied per-group in the K loop,
    # so epilogue can skip weight scale multiplication (uses 1.0 for sw).

    _is_gfx950 = "gfx95" in gpu_arch
    if gpu_arch != "gfx942" and gpu_arch != "gfx950":
        raise ValueError(
            "FlyDSL nvfp4_bf16 MoE kernels are supported only on gfx942/gfx950, "
            f"got {gpu_arch}."
        )
    use_gfx950_cvt = _is_gfx950

    # Split-K validation
    _is_splitk = k_batch > 1
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert model_dim % k_batch == 0, (
            f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        )
        assert _k_per_batch % tile_k == 0, (
            f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"
        )
        # The ping-pong K-loop requires an even number of K tiles (>=4).
        _k_tiles = _k_per_batch // tile_k
        assert _k_tiles >= 4 and _k_tiles % 2 == 0, (
            f"K_per_batch/tile_k={_k_tiles} must be even and >=4 for "
            "the ping-pong pipeline. "
            f"Try a different k_batch (model_dim={model_dim}, tile_k={tile_k})."
        )
    else:
        _k_per_batch = model_dim
        _k_tiles = _k_per_batch // tile_k
        if _k_tiles < 2 or _k_tiles % 2 != 0:
            raise ValueError(
                "FlyDSL nvfp4_bf16 stage1 requires an even number of K tiles "
                f">= 2, got model_dim/tile_k={_k_tiles} "
                f"(model_dim={model_dim}, tile_k={tile_k})."
            )

    mfma_f32_bf16_k16 = getattr(rocdl, "mfma_f32_16x16x16bf16_1k", None) or getattr(
        rocdl, "mfma_f32_16x16x16_bf16_1k", None
    )
    if mfma_f32_bf16_k16 is None:
        raise AttributeError("BF16 K16 MFMA op not found")

    # gfx950: use 16x16x32 MFMA for f16/bf16 (K=32 per MFMA, vs K=16 on gfx942).
    # Check if K=32 MFMA supports the (result_type, operands_list) calling convention.
    _has_k32_mfma_compat = False
    if _is_gfx950:
        import inspect

        _k32_fn = rocdl.mfma_f32_16x16x32_bf16
        try:
            _k32_sig = inspect.signature(_k32_fn)
            _k32_params = list(_k32_sig.parameters.keys())
            # Compatible if second param is "operands" (list-based API)
            _has_k32_mfma_compat = (
                len(_k32_params) >= 2 and _k32_params[1] == "operands"
            )
        except (ValueError, TypeError):
            _has_k32_mfma_compat = False
    _use_mfma_k32 = _is_gfx950 and _has_k32_mfma_compat

    ir.ShapedType.get_dynamic_size()
    # W is packed int4 for W4A8/W4A16/W4A_FP8: 2 values per byte.
    w_nbytes = (experts * (2 * inter_dim) * model_dim) // 2
    sw_nbytes = experts * (2 * inter_dim) * num_groups

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, "
            f"elem_bytes={elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use direct buffer_load for smaller widths.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    lds_stride = tile_k
    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = True
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    # Split-K uses f32 atomic CShuffle regardless of out_dtype, so skip this check.
    if out_dtype != "f16" and use_cshuffle_epilog and not _is_splitk:
        raise ValueError(
            "stage1 cshuffle epilog currently supports only "
            "f16 output (out_dtype='f16')"
        )

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    # IMPORTANT: module name participates in FlyDSL's compile cache key.
    # Keep an explicit ABI tag so signature changes can't
    # accidentally reuse an old binary.
    _gs_tag = f"_g{group_size}"
    _split_k_tag = f"_splitk{k_batch}" if _is_splitk else ""
    (
        f"mfma_moe1_{in_dtype}_{out_dtype}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_gs_tag}{_split_k_tag}"
        # _abi4: also masks sentinel token ids on loads to avoid illegal address faults
        f"_abi4"
    ).replace("-", "_")

    # ── LDS sizing (pure Python; no MLIR Context needed) ─────────────────────
    # Reuse the same LDS bytes for both:
    # - ping-pong X tiles (2 * tile_m * lds_stride bytes)
    # - optional epilogue CShuffle tile
    #       (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    # Split-K requires CShuffle epilogue (atomic adds via store_pair callback)
    if _is_splitk:
        _use_cshuffle_epilog = True
    # bf16 split-K: use bf16 atomics (halves bandwidth,
    # gfx950 has buffer_atomic_pk_add_bf16).
    # Other dtypes keep f32 for precision.
    _splitk_use_bf16 = _is_splitk
    _cshuffle_elem_bytes = 2 if (not _is_splitk or _splitk_use_bf16) else 4
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(elem_bytes)
    lds_out_bytes = (
        _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes)
    lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

    lds_alloc_bytes = int(lds_total_elems) * int(elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    @flyc.kernel
    def moe_gemm1(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_global_scale: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_max_token_ids: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        inter_in = arith.index_cast(T.index, i32_inter_in)
        k_in = arith.index_cast(T.index, i32_k_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        # i32 versions for layout construction (fly.make_shape requires i32/i64)
        tokens_i32_v = i32_tokens_in
        k_i32_v = i32_k_in
        x_elem = T.bf16
        # Packed 4-bit weights are stored as bytes and unpacked in-kernel.
        w_elem = T.i8
        vec16_elems = 16 if elem_bytes == 1 else 8
        vec8_elems = 8 if elem_bytes == 1 else 4
        vec8_x = T.vec(vec8_elems, x_elem)
        vec16_x = T.vec(vec16_elems, x_elem)

        _ptr_buffer_resource = functools.partial(
            _make_ptr_buffer_resource, arith=arith, buffer_ops=buffer_ops
        )

        def silu(x):
            # device fast path:
            #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
            #   sig = rcp(1 + emu)                   -> v_rcp_f32
            #   y = x * sig
            #
            # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
            # sequences that introduce extra compares/cndmasks.
            t = x * (-1.4426950408889634)  # -log2(e)
            emu = rocdl.exp2(T.f32, t)
            den = 1.0 + emu
            sig = rocdl.rcp(T.f32, den)
            return x * sig

        acc_init = arith.constant_vector(0.0, T.f32x4)

        # Layouts (use i32 values; fly.make_shape requires i32/i64, not index)
        fx.make_layout((tokens_i32_v, k_i32_v), stride=(k_i32_v, 1))

        # B preshuffle layout: match GEMM test helper exactly.
        c_n_total = arith.index(experts * (2 * inter_dim))
        # For packed 4-bit weights, kpack_bytes=8.
        kpack_bytes = 8
        w_elem_bytes = 1
        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=c_n_total,
            c_k=k_in,
            kpack_bytes=kpack_bytes,
            elem_bytes=w_elem_bytes,
        )
        layout_b = b_layout.layout_b
        (k_in * arith.index(int(elem_bytes))) // fx.Index(64)

        shape_lds = fx.make_shape(tile_m, tile_k)
        stride_lds = fx.make_stride(lds_stride, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        tx = gpu.thread_id("x")
        # Align with Aiter launch mapping (NSwizzle==false):
        # - blockIdx.x -> N dimension (tile along inter_dim)
        # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
        by = gpu.block_id("x")  # tile along inter_dim
        bx = gpu.block_id("y")  # tile along sorted M

        if const_expr(_is_splitk):
            bz = gpu.block_id("z")  # K-batch id
            k_base_idx = bz * arith.index(_k_per_batch)
        else:
            k_base_idx = arith.index(0)

        # Block validity: compute as early as possible so invalid blocks skip
        # all buffer-resource setup, LDS pointer math, and gmem prefetch work.
        bx_m = bx * fx.Index(tile_m)
        maxids_rsrc = _ptr_buffer_resource(arg_max_token_ids, fx.Index(4))
        max_token_id_i32 = buffer_ops.buffer_load(
            maxids_rsrc, fx.Index(0), vec_width=1, dtype=T.i32
        )
        bx_m_i32 = arith.index_cast(T.i32, bx_m)
        blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, max_token_id_i32)
        # Common constants/atoms (hoisted): keep IR small like GEMM.
        # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
        k_blocks16 = arith.index(tile_k_bytes // 16)
        layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

        # Everything below is gated by `blk_valid` to avoid doing buffer-resource
        # setup and gmem work for padding blocks.
        _if_blk = scf.IfOp(blk_valid)
        with _if_then(_if_blk):
            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                T.bf16,
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias LDS bytes for optional CShuffle epilogue.
            # bf16 split-K uses bf16 (2B); other split-K uses f32 (4B);
            # normal uses f16/bf16 (2B).
            _lds_out_elem_type = T.bf16
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    _lds_out_elem_type,
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # Buffer resources: for dynamic memrefs,
            # provide `num_records_bytes` explicitly so hardware OOB behavior
            # is stable (otherwise it falls back to a large max size).
            c_topk = fx.Index(topk)

            # X: [tokens, k] bytes = tokens*k*elem_bytes
            x_nbytes_idx = tokens_in * k_in * arith.index(int(elem_bytes))
            x_rsrc = _ptr_buffer_resource(arg_x, x_nbytes_idx)

            w_rsrc = _ptr_buffer_resource(arg_w, w_nbytes)

            # OUT: normal=[tokens, topk, inter] f16/bf16,
            #      split-K=[tokens*topk, 2*inter] f32 (or bf16 for bf16 split-K)
            out_elem_bytes = 4 if (_is_splitk and not _splitk_use_bf16) else 2
            if const_expr(_is_splitk):
                out_nbytes_idx = (
                    tokens_in * c_topk * inter_in * fx.Index(2 * out_elem_bytes)
                )
            else:
                out_nbytes_idx = (
                    tokens_in * c_topk * inter_in * fx.Index(out_elem_bytes)
                )
            out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes_idx)

            sw_rsrc = _ptr_buffer_resource(arg_scale_w, sw_nbytes)
            gs_rsrc = _ptr_buffer_resource(arg_global_scale, fx.Index(experts * 4))

            sorted_nbytes_idx = size_expert_ids_in * fx.Index(tile_m) * fx.Index(4)
            sorted_rsrc = _ptr_buffer_resource(arg_sorted_token_ids, sorted_nbytes_idx)
            sorted_w_rsrc = _ptr_buffer_resource(arg_sorted_weights, sorted_nbytes_idx)

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            expert_rsrc = _ptr_buffer_resource(
                arg_expert_ids, size_expert_ids_in * fx.Index(4)
            )

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            inter2_idx = arith.index(2 * inter_dim)
            expert_off_idx = expert_idx * inter2_idx  # index
            global_scale_f32 = buffer_ops.buffer_load(
                gs_rsrc,
                arith.index_cast(T.i32, expert_idx),
                vec_width=1,
                dtype=T.f32,
            )

            if const_expr(bytes_per_thread_x % 16 != 0):
                raise ValueError(
                    "bytes_per_thread_x must be divisible by 16 for BF16 NVFP4"
                )
            x_load_bytes = 16
            num_x_loads = bytes_per_thread_x // x_load_bytes
            chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)

            c_k_div4 = (k_in * arith.index(int(elem_bytes))) // fx.Index(4)
            c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
            fx.make_layout((tokens_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1))
            tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
            layout_x_tile_div4 = fx.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )
            c_chunk_i32 = fx.Index(chunk_i32)
            tx_i32_base = tx * c_chunk_i32
            mask24 = fx.Int32(0xFFFFFF)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)
            topk_i32 = fx.Int32(topk)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                    chunk_i32=chunk_i32,
                )

            # decode token once (per thread's M-slice) and build a base row offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                # NOTE: rows beyond `num_valid_ids` can contain garbage
                # (within the allocated buffer).
                # That's OK as long as we never use an
                # out-of-range token id to index X.
                fused_i = buffer_ops.buffer_load(
                    sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                )
                t_raw = fused_i & mask24
                # NOTE: aiter moe_sorting uses sentinel token_id == tokens
                # for padding. Do NOT rely on buffer OOB semantics for X loads;
                # explicitly mask to a safe row.
                t_valid_i32 = arith.cmpi(arith.CmpIPredicate.ult, t_raw, tokens_i32)
                t_idx = arith.index_cast(T.index, t_raw)
                t_safe = t_valid_i32.select(t_idx, fx.Index(0))
                x_row_base_div4.append(t_safe * c_k_div4)

            vec4_x = T.vec(4, x_elem)

            load_x = functools.partial(
                load_bf16_x,
                buffer_ops=buffer_ops,
                vector=vector,
                x_elem=x_elem,
                x_rsrc=x_rsrc,
                vec16_elems=vec16_elems,
            )

            def load_x_tile(base_k):
                """Prefetch the per-thread X tile portion (gmem -> regs)
                for a given K base (in elements)."""
                base_k_div4 = (base_k * arith.index(int(elem_bytes))) // fx.Index(4)
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_vec = load_x(idx_i32)
                    if const_expr(x_load_bytes == 16):
                        parts.append(vector.bitcast(T.i32x4, x_vec))
                    elif const_expr(x_load_bytes == 8):
                        parts.append(x_vec)
                    else:
                        parts.append(x_vec)
                return parts

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = fx.idx2crd(fx.Int32(tx), layout_tx_wave_lane)
            wave_id = fx.get(coord_wl, 0)
            lane_id = fx.get(coord_wl, 1)
            coord_l16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
            lane_div_16 = fx.get(coord_l16, 0)
            lane_mod_16 = fx.get(coord_l16, 1)

            # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col
            # base is lane_div_16 * a_kpack_elems.
            # A-side kpack is always 16 bytes (activation elements);
            # B-side kpack_bytes may differ (e.g. 8 for int4 weights),
            # but that only affects B preshuffle.
            row_a_lds = lane_mod_16
            a_kpack_elems = 16 // elem_bytes
            col_offset_base = lane_div_16 * arith.index(int(a_kpack_elems))
            col_offset_base_bytes = (
                col_offset_base
                if elem_bytes == 1
                else (col_offset_base * arith.index(int(elem_bytes)))
            )

            # Dynamic N tiling within block (same as existing kernels)
            by_n = by * fx.Index(tile_n)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = fx.Index(n_per_wave)
            wave_mod_4 = wave_id % fx.Index(4)
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            inter_idx = fx.Index(inter_dim)
            c_n_total // fx.Index(16)
            c_n0_static = experts * (2 * inter_dim) // 16
            layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.index(ni * 16)
                col_g = by_n + n_tile_base
                col_g = col_g + offset
                col_g = col_g + lane_mod_16
                col_g_list.append(col_g)

                row_gate = expert_off_idx + col_g
                row_up = row_gate + inter_idx

                coord_gate = fx.idx2crd(fx.Int32(row_gate), layout_n_blk_intra)
                n_blk_gate.append(fx.get(coord_gate, 0))
                n_intra_gate.append(fx.get(coord_gate, 1))

                coord_up = fx.idx2crd(fx.Int32(row_up), layout_n_blk_intra)
                n_blk_up.append(fx.get(coord_up, 0))
                n_intra_up.append(fx.get(coord_up, 1))

            m_repeat = tile_m // 16
            k_unroll = tile_k_bytes // 64  # K64-byte micro-step (2x MFMA)

            load_b_tile = functools.partial(
                load_nvfp4_b_tile,
                buffer_ops=buffer_ops,
                arith=arith,
                vector=vector,
                arg_w=arg_w,
                w_rsrc=w_rsrc,
                layout_b=layout_b,
                lane_div_16=lane_div_16,
                w_elem=w_elem,
                sw_rsrc=sw_rsrc,
                expert_idx=expert_idx,
                num_groups=num_groups,
                n_per_expert=2 * inter_dim,
                kpack_bytes=kpack_bytes,
                k_unroll=k_unroll,
                num_acc_n=num_acc_n,
            )

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            store_x_tile_to_lds = functools.partial(
                _store_x_tile_to_lds,
                arith=arith,
                vector=vector,
                x_row_local=x_row_local,
                x_col_local_i32=x_col_local_i32,
                num_x_loads=num_x_loads,
                x_load_bytes=x_load_bytes,
                lds_x=lds_x,
                vec16_x=vec16_x,
                vec8_x=vec8_x,
                vec4_x=vec4_x,
                layout_lds=layout_lds,
                k_blocks16=k_blocks16,
                elem_bytes=elem_bytes,
            )

            lds_load_packs_k64 = functools.partial(
                lds_load_bf16_k64,
                arith=arith,
                vector=vector,
                k_blocks16=k_blocks16,
                layout_lds=layout_lds,
                vec16_x=vec16_x,
                lds_x=lds_x,
            )

            def compute_tile(
                acc_gate_in,
                acc_up_in,
                b_gate_tile_in,
                b_up_tile_in,
                lds_base,
                *,
                prefetch_epilogue: bool = False,
                a0_prefetch=None,
            ):
                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)
                mfma_res_ty = T.f32x4
                if const_expr(_use_mfma_k32):
                    mfma_fn = rocdl.mfma_f32_16x16x32_bf16
                else:
                    mfma_fn = mfma_f32_bf16_k16

                mfma_k64 = functools.partial(
                    _mfma_k64,
                    mfma_fn=mfma_fn,
                    mfma_res_ty=mfma_res_ty,
                    use_mfma_k32=_use_mfma_k32,
                    vector=vector,
                )

                # Optional: prefetch epilogue scales while we are about
                # to run the last MFMA tile, matching the preshuffle GEMM
                # pattern of overlapping scale loads with MFMA.
                epilogue_pf = None
                if const_expr(prefetch_epilogue and not use_inloop_w_scale):
                    expert_off_pf = expert_off_idx
                    sw_gate_pf = []
                    sw_up_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_gate_idx = expert_off_pf + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_pf.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_gate_idx, vec_width=1, dtype=T.f32
                            )
                        )
                        sw_up_pf.append(
                            fx.Float32(1.0)
                            if not needs_scale_w
                            else buffer_ops.buffer_load(
                                sw_rsrc, row_up_idx, vec_width=1, dtype=T.f32
                            )
                        )
                    epilogue_pf = (sw_gate_pf, sw_up_pf)

                for ku in range_constexpr(k_unroll):
                    b_gate_raw = b_gate_tile_in[ku]
                    b_up_raw = b_up_tile_in[ku]
                    ki64 = arith.index(ku * 64)
                    col_base = col_offset_base_bytes + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.index(mi * 16)
                        curr_row_a_lds = row_a_lds + mi_val

                        if const_expr(
                            (a0_prefetch is not None) and (ku == 0) and (mi == 0)
                        ):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(
                                curr_row_a_lds, col_base, lds_base
                            )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            packed_g, sc_g = b_gate_raw[ni]
                            bg0, bg1 = unpack_b_nvfp4(
                                packed_g,
                                sc_g,
                                arith,
                                vector,
                                use_gfx950_cvt=use_gfx950_cvt,
                            )
                            gate_list[acc_idx] = mfma_k64(
                                gate_list[acc_idx], a0, a1, bg0, bg1
                            )
                            packed_u, sc_u = b_up_raw[ni]
                            bu0, bu1 = unpack_b_nvfp4(
                                packed_u,
                                sc_u,
                                arith,
                                vector,
                                use_gfx950_cvt=use_gfx950_cvt,
                            )
                            up_list[acc_idx] = mfma_k64(
                                up_list[acc_idx], a0, a1, bu0, bu1
                            )
                return gate_list, up_list, epilogue_pf

            # 2-stage pipeline (ping-pong LDS + B tile prefetch)
            lds_tile_elems = arith.index(tile_m * lds_stride)
            lds_base_cur = fx.Index(0)
            lds_base_nxt = lds_tile_elems

            # Optional scheduler hints (copied from tuned GEMM);
            # can be disabled via env.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                rocdl.sched_barrier(0)
                return

            # Prologue: prefetch tile0, store to LDS(cur), sync.
            k0 = k_base_idx
            x_regs0 = load_x_tile(k0)
            b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
            b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up)
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            # Loop-carried ping/pong state.
            lds_base_pong = lds_base_cur  # current/compute
            lds_base_ping = lds_base_nxt  # next/load+store

            # Cross-tile A0 LDS prefetch (default-on): prefetch the
            # first A-pack (K64) for the tile we are about to compute
            # from LDS, to overlap with upcoming VMEM.
            a0_prefetch_pong = lds_load_packs_k64(
                row_a_lds, col_offset_base_bytes, lds_base_pong
            )

            # Ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
            # Uses scf.for with loop-carried accumulators,
            # B-tile prefetch, and A0 LDS prefetch.
            arith.index(tile_k * 2)
            c_tile_k = arith.index(tile_k)
            total_tiles = int(_k_per_batch) // int(tile_k)
            pair_iters = max((total_tiles - 2) // 2, 0)

            # Each NVFP4 entry carries one packed weight dword and one block scale.
            _fields_per_ku = 2
            _vals_per_b_tile = k_unroll * _fields_per_ku * num_acc_n

            _flatten_b_tile = flatten_nvfp4_b_tile
            _unflatten_b_tile = functools.partial(
                unflatten_nvfp4_b_tile,
                k_unroll=k_unroll,
                num_acc_n=num_acc_n,
            )

            init_state = (
                list(acc_gate)
                + list(acc_up)
                + _flatten_b_tile(b_gate_cur)
                + _flatten_b_tile(b_up_cur)
                + list(a0_prefetch_pong)
            )

            _n_acc = m_repeat * num_acc_n
            _p_bg = 2 * _n_acc
            _p_bu = _p_bg + _vals_per_b_tile
            _p_a0 = _p_bu + _vals_per_b_tile

            for pair_iv, state in range(  # type: ignore[call-overload]
                0, pair_iters, 1, init=init_state
            ):
                _ag = list(state[:_n_acc])
                _au = list(state[_n_acc:_p_bg])
                _bg = _unflatten_b_tile(list(state[_p_bg:_p_bu]))
                _bu = _unflatten_b_tile(list(state[_p_bu:_p_a0]))
                _a0pf = (state[_p_a0], state[_p_a0 + 1])

                k_iv = k_base_idx + pair_iv * (c_tile_k + c_tile_k)

                # ---- stage 0: prefetch+store ping, compute pong ----
                next_k1 = k_iv + c_tile_k
                x_regs_ping = load_x_tile(next_k1)
                _bg_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                _bu_ping = load_b_tile(next_k1, n_blk_up, n_intra_up)

                _ag, _au, _ = compute_tile(
                    _ag, _au, _bg, _bu, lds_base_pong, a0_prefetch=_a0pf
                )
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()

                _a0pf_ping = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_ping
                )

                # ---- stage 1: prefetch+store pong, compute ping ----
                next_k2 = k_iv + c_tile_k + c_tile_k
                x_regs_pong = load_x_tile(next_k2)
                _bg_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                _bu_next = load_b_tile(next_k2, n_blk_up, n_intra_up)

                _ag, _au, _ = compute_tile(
                    _ag,
                    _au,
                    _bg_ping,
                    _bu_ping,
                    lds_base_ping,
                    a0_prefetch=_a0pf_ping,
                )
                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

                _a0pf_new = lds_load_packs_k64(
                    row_a_lds, col_offset_base_bytes, lds_base_pong
                )

                loop_results = yield (
                    list(_ag)
                    + list(_au)
                    + _flatten_b_tile(_bg_next)
                    + _flatten_b_tile(_bu_next)
                    + list(_a0pf_new)
                )

            # After scf.for: extract final state from yielded results.
            SmemPtr._view_cache = None
            if pair_iters > 0:
                acc_gate = list(loop_results[:_n_acc])
                acc_up = list(loop_results[_n_acc:_p_bg])
                b_gate_cur = _unflatten_b_tile(list(loop_results[_p_bg:_p_bu]))
                b_up_cur = _unflatten_b_tile(list(loop_results[_p_bu:_p_a0]))
                a0_prefetch_pong = (loop_results[_p_a0], loop_results[_p_a0 + 1])
            k_tail1 = k_base_idx + arith.index(_k_per_batch - tile_k)
            x_regs_ping = load_x_tile(k_tail1)
            b_gate_ping = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
            b_up_ping = load_b_tile(k_tail1, n_blk_up, n_intra_up)

            acc_gate, acc_up, _ = compute_tile(
                acc_gate,
                acc_up,
                b_gate_cur,
                b_up_cur,
                lds_base_pong,
                a0_prefetch=a0_prefetch_pong,
            )
            a0_prefetch_pong = None
            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
            hot_loop_scheduler()
            gpu.barrier()

            # Cross-tile prefetch for the final ping tile.
            a0_prefetch_ping = lds_load_packs_k64(
                row_a_lds, col_offset_base_bytes, lds_base_ping
            )

            # Epilogue: compute last tile with epilogue scale prefetch to
            # overlap loads with MFMA.
            acc_gate, acc_up, epilogue_pf = compute_tile(
                acc_gate,
                acc_up,
                b_gate_ping,
                b_up_ping,
                lds_base_ping,
                prefetch_epilogue=True,
                a0_prefetch=a0_prefetch_ping,
            )

            # Store epilogue to out[t, slot, inter]
            tokens_i32_v = tokens_i32
            topk_i32_v = topk_i32
            inter_i32_v = fx.Int32(inter_dim)
            mask24_i32 = fx.Int32(0xFFFFFF)

            sw_gate_vals = [global_scale_f32] * num_acc_n
            sw_up_vals = [global_scale_f32] * num_acc_n

            # Epilogue hoists to keep IR + Python build time small:
            col_i32_list = []
            for ni in range_constexpr(num_acc_n):
                col_i32_list.append(arith.index_cast(T.i32, col_g_list[ni]))

            lane_div_16 * fx.Index(4)
            inter_i32_local = inter_i32_v

            # Uses EVec=4 (buffer store "x4" of fp16 elements).
            use_cshuffle_epilog_flag = _use_cshuffle_epilog

            # ─── Split-K epilogue: two-pass gate/up with atomic fadd ───
            # bf16 split-K uses bf16 atomics; other dtypes use f32 atomics.
            if const_expr(_is_splitk):
                if const_expr(lds_out is None):
                    raise RuntimeError("Split-K epilogue requires lds_out (CShuffle)")

                _has_buffer_atomic_bf16_s1 = str(gpu_arch).startswith(
                    ("gfx95", "gfx12")
                )
                _needs_global_atomic_bf16_s1 = (
                    _splitk_use_bf16 and not _has_buffer_atomic_bf16_s1
                )

                out_base_idx = arith.index_cast(T.index, fx.ptrtoint(arg_out))
                _split_k_out_row_stride = (
                    inter_dim * 2 * out_elem_bytes
                )  # bytes per row
                _split_k_e_vec = 2  # vec2 for atomic fadd (f32 or bf16)

                # Mutable slot: 0 for gate pass, inter_dim for up pass
                _split_k_n_offset = [0]

                # Mutable slots for two-pass gate/up selection
                _split_k_acc = [acc_gate]
                _split_k_sw_vals = [sw_gate_vals]

                _splitk_lds_elem = T.bf16 if _splitk_use_bf16 else T.f32
                _splitk_lds_align = 2 if _splitk_use_bf16 else 4

                def write_row_to_lds_splitk(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    """Write scaled partial sums to LDS (no silu, no doweight)."""
                    _acc = _split_k_acc[0]
                    _sw = _split_k_sw_vals[0]
                    # Load per-row scale_x (sx) — same logic as normal epilogue.
                    sx = fx.Float32(1.0)
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            _acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
                        v = v * sx * _sw[ni]
                        if _splitk_use_bf16:
                            v = arith.trunc_f(T.bf16, v)
                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(T.vec(1, _splitk_lds_elem), [v])
                        vector.store(
                            v1, lds_out, [lds_idx], alignment=_splitk_lds_align
                        )

                def precompute_row_splitk(*, row_local, row):
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                    t_idx = arith.index_cast(T.index, t2)
                    s_idx = arith.index_cast(T.index, s2)
                    ts_idx = t_idx * arith.index(topk) + s_idx
                    if const_expr(
                        _splitk_use_bf16 and not _needs_global_atomic_bf16_s1
                    ):
                        # For buffer atomics: compute relative byte offset
                        # from buffer base
                        row_byte_off = ts_idx * arith.index(_split_k_out_row_stride)
                        return (row_byte_off, t_ok)
                    else:
                        # For global atomics: compute absolute address
                        row_byte_base = out_base_idx + ts_idx * arith.index(
                            _split_k_out_row_stride
                        )
                        return (row_byte_base, t_ok)

                _splitk_zero_i32 = [fx.Int32(0) if _splitk_use_bf16 else None]

                def store_pair_splitk(
                    *, row_local, row, row_ctx, col_pair0, col_g0, frag
                ):
                    row_byte_ctx = row_ctx
                    col_idx = col_g0 + arith.index(_split_k_n_offset[0])
                    byte_off_col = col_idx * arith.index(out_elem_bytes)
                    if const_expr(_splitk_use_bf16):
                        _z = _splitk_zero_i32[0]
                        if const_expr(_needs_global_atomic_bf16_s1):
                            # gfx942: global atomicrmw fadd for bf16
                            ptr_addr_idx = row_byte_ctx + byte_off_col
                            out_ptr = buffer_ops.create_llvm_ptr(
                                ptr_addr_idx, address_space=1
                            )
                            out_ptr_v = (
                                out_ptr._value
                                if hasattr(out_ptr, "_value")
                                else out_ptr
                            )
                            frag_v = frag._value if hasattr(frag, "_value") else frag
                            llvm.AtomicRMWOp(
                                llvm.AtomicBinOp.fadd,
                                out_ptr_v,
                                frag_v,
                                llvm.AtomicOrdering.monotonic,
                                syncscope="agent",
                                alignment=_split_k_e_vec * out_elem_bytes,
                            )
                        else:
                            # gfx950+: buffer_atomic_pk_add_bf16
                            byte_off_i32 = arith.index_cast(
                                T.i32, row_byte_ctx + byte_off_col
                            )
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                frag,
                                out_rsrc,
                                byte_off_i32,
                                _z,
                                _z,
                            )
                    else:
                        # f32 atomic: global atomicrmw fadd
                        ptr_addr_idx = row_byte_ctx + byte_off_col
                        out_ptr = buffer_ops.create_llvm_ptr(
                            ptr_addr_idx, address_space=1
                        )
                        out_ptr_v = (
                            out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                        )
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_split_k_e_vec * out_elem_bytes,
                        )

                _cshuffle_nlane_splitk = min(32, tile_n // _split_k_e_vec)
                _splitk_frag_elem = (
                    ir.BF16Type.get() if _splitk_use_bf16 else ir.F32Type.get()
                )

                # Pass 1: gate (offset=0)
                _split_k_acc[0] = acc_gate
                _split_k_sw_vals[0] = sw_gate_vals
                _split_k_n_offset[0] = 0
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_split_k_e_vec,
                    cshuffle_nlane=_cshuffle_nlane_splitk,
                    block_size=total_threads,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=_splitk_frag_elem,
                    write_row_to_lds=write_row_to_lds_splitk,
                    precompute_row=precompute_row_splitk,
                    store_pair=store_pair_splitk,
                )

                gpu.barrier()

                # Pass 2: up (offset=inter_dim)
                _split_k_acc[0] = acc_up
                _split_k_sw_vals[0] = sw_up_vals
                _split_k_n_offset[0] = inter_dim
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_split_k_e_vec,
                    cshuffle_nlane=_cshuffle_nlane_splitk,
                    block_size=total_threads,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=_splitk_frag_elem,
                    write_row_to_lds=write_row_to_lds_splitk,
                    precompute_row=precompute_row_splitk,
                    store_pair=store_pair_splitk,
                )
                return

            if const_expr(use_cshuffle_epilog_flag):
                if const_expr(lds_out is None):
                    raise RuntimeError(
                        "CShuffle epilogue enabled but lds_out is "
                        "not allocated/aliased."
                    )

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    sx = fx.Float32(1.0)

                    # Sorted weight aligned with `row`
                    # (matches aiter moe_sorting output).
                    if const_expr(doweight_stage1):
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                        )

                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        sw_gate = sw_gate_vals[ni]
                        sw_up = sw_up_vals[ni]

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(
                            acc_gate[acc_idx],
                            static_position=[ii],
                            dynamic_position=[],
                        )
                        vu = vector.extract(
                            acc_up[acc_idx],
                            static_position=[ii],
                            dynamic_position=[],
                        )

                        vg = vg * sx * sw_gate
                        vu = vu * sx * sw_up

                        y = silu(vg) * vu
                        if const_expr(doweight_stage1):
                            y = y * tw
                        y16 = arith.trunc_f(T.f16, y)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(T.vec(1, T.f16), [y16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    return (t2 * topk_i32_v + s2) * inter_i32_local

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    # Guard against sentinel token ids (t == tokens) produced by
                    # aiter moe_sorting padding.
                    # OOB buffer stores are not guaranteed to be safe on all paths,
                    # so predicate explicitly.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                    _if_valid = scf.IfOp(t_valid)
                    with _if_then(_if_valid):
                        idx0 = row_ctx
                        col_i32 = arith.index_cast(T.i32, col_g0)
                        idx_out = idx0 + col_i32
                        # Vectorized fp16 store (EVec=4).
                        buffer_ops.buffer_store(frag, out_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=True,
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=4,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )
                return

            def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                # `row` is the sorted-row index (bx_m + row_in_tile).
                # Block-level early-exit already guards `bx_m` range.
                # Here we rely on buffer OOB semantics for any tail rows.
                fused2 = buffer_ops.buffer_load(
                    sorted_rsrc, row, vec_width=1, dtype=T.i32
                )
                t2_raw = fused2 & mask24_i32
                s2_raw = fused2 >> 24
                t2 = t2_raw
                s2 = s2_raw
                t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                # Do NOT rely on buffer OOB semantics for scale loads; explicitly mask.
                sx0 = fx.Float32(1.0)
                sx = sx0
                arith.constant(0.0, type=out_mlir())

                # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                # Sorted weight aligned with `row` (matches aiter moe_sorting output).
                if const_expr(doweight_stage1):
                    tw = buffer_ops.buffer_load(
                        sorted_w_rsrc, row, vec_width=1, dtype=T.f32
                    )

                _if_valid = scf.IfOp(t_valid)
                with _if_then(_if_valid):
                    for ni in range_constexpr(num_acc_n):
                        col_i32 = col_i32_list[ni]
                        sw_gate = sw_gate_vals[ni]
                        sw_up = sw_up_vals[ni]

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(
                            acc_gate[acc_idx],
                            static_position=[ii],
                            dynamic_position=[],
                        )
                        vu = vector.extract(
                            acc_up[acc_idx],
                            static_position=[ii],
                            dynamic_position=[],
                        )

                        vg = vg * sx * sw_gate
                        vu = vu * sx * sw_up

                        y = silu(vg) * vu
                        if const_expr(doweight_stage1):
                            y = y * tw
                        y = arith.trunc_f(out_mlir(), y)
                        idx_out0 = idx0 + col_i32
                        buffer_ops.buffer_store(y, out_rsrc, idx_out0)

            mfma_epilog(
                use_cshuffle=False,
                arith=arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=_stage1_store_row,
            )

    # ── Host launcher (flyc.jit + .launch) ────────────────────────────────
    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w: fx.Pointer,
        arg_scale_x: fx.Pointer,
        arg_scale_w: fx.Pointer,
        arg_global_scale: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_max_token_ids: fx.Pointer,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = inter_in // fx.Index(tile_n)
        gy = size_expert_ids_in

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_global_scale,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, k_batch),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm1
