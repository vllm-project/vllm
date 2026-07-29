# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from vllm.triton_utils import aggregate, gl, gluon, tl

_INV_LN2_VALUE = 1.4426950408889634
_INV_LN2 = tl.constexpr(_INV_LN2_VALUE)
_LN2_VALUE = 0.6931471805599453
_LN2 = tl.constexpr(_LN2_VALUE)
_PROPAGATE_NAN_ALL = gl.constexpr(tl.PropagateNan.ALL)


@gluon.jit
def maximum(a, b, propagate_nan: gl.constexpr = _PROPAGATE_NAN_ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


@gluon.constexpr_function
def attention_layouts(head_dim, block_n, is_fp8, dtype, num_warps, instr_shape):
    mfma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=instr_shape,
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    qk_layout = mfma
    pv_layout = mfma
    # qk_kw is derived from a 128-bit load / dtype bitwidth; pv_kw is tuned.
    qk_kw = 16 if is_fp8 else 8
    pv_kw = 8 if is_fp8 else 4
    q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
    k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
    p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
    v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
    # load_vec = elems/lane (dtype-dependent, == qk_kw); load_threads span HEAD_DIM.
    load_vec = 16 if is_fp8 else 8
    load_threads = head_dim // load_vec
    load_layout = gl.BlockedLayout(
        [1, load_vec], [64 // load_threads, load_threads], [num_warps, 1], [1, 0]
    )
    # store_vec is always 16-bit (128 / 16 = 8) regardless of input dtype.
    store_vec = 8
    store_threads = head_dim // store_vec
    store_layout = gl.BlockedLayout(
        [1, store_vec], [64 // store_threads, store_threads], [num_warps, 1], [1, 0]
    )
    # Take only the built-in's padding, not its swizzle: the swizzle scatters
    # head_dim across banks, making the LDS write stride non-constant, so it can't
    # lower through async_copy's affine [1, 0] load. Padding-only is affine and
    # DMA-legal.
    # TODO(perf): to also use the swizzle, co-design a matched load layout so the
    # DMA stays legal.
    k_api = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        k_layout, [block_n, head_dim], dtype, is_k_contig=True
    )
    v_api = gl.amd.cdna4.compute_efficient_padded_shared_layout(
        v_layout, [block_n, head_dim], dtype, is_k_contig=False
    )
    assert k_api is not None and v_api is not None, (
        "no CDNA4 padded shared layout for this operand/dtype"
    )
    k_pairs = list(k_api.interval_padding_pairs)
    v_pairs = list(v_api.interval_padding_pairs)
    assert len(k_pairs) == 1 and len(v_pairs) == 1, (
        "expected a single interval padding pair from the built-in"
    )
    k_smem_layout = gl.PaddedSharedLayout.with_identity_for(
        [[int(k_pairs[0][0]), int(k_pairs[0][1])]], [block_n, head_dim], [1, 0]
    )
    v_smem_layout = gl.PaddedSharedLayout.with_identity_for(
        [[int(v_pairs[0][0]), int(v_pairs[0][1])]], [block_n, head_dim], [1, 0]
    )
    return (
        qk_layout,
        pv_layout,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        load_layout,
        store_layout,
        k_smem_layout,
        v_smem_layout,
    )


@aggregate
class InputStrides:
    stride_t: gl.constexpr
    stride_h: gl.constexpr
    stride_d: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, stride_t, stride_h, stride_d):
        self.stride_t = gl.constexpr(stride_t)
        self.stride_h = gl.constexpr(stride_h)
        self.stride_d = gl.constexpr(stride_d)

    @gluon.jit
    def offsets(self, token, head, dim):
        return (token * self.stride_t + head * self.stride_h + dim * self.stride_d).to(
            gl.int32
        )


@aggregate
class PagedKVStrides:
    stride_b: gl.constexpr
    stride_p: gl.constexpr
    stride_h: gl.constexpr
    stride_d: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, stride_b, stride_p, stride_h, stride_d):
        self.stride_b = gl.constexpr(stride_b)
        self.stride_p = gl.constexpr(stride_p)
        self.stride_h = gl.constexpr(stride_h)
        self.stride_d = gl.constexpr(stride_d)

    @gluon.jit
    def offsets(self, page, token, head, dim):
        # KV pools can exceed the 32-bit buffer-offset range. Keep the page
        # arithmetic in int64, matching the original kernel's large-cache path.
        return (
            page.to(gl.int64) * self.stride_b
            + token.to(gl.int64) * self.stride_p
            + head * self.stride_h
            + dim * self.stride_d
        )
