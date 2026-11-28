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
