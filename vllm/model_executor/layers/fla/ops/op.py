# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

from vllm.triton_utils import tl, tldevice, triton

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    div = tldevice.fast_dividef
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:

    @triton.jit
    def div_normal(x, y):
        return x / y

    div = div_normal
    exp = tl.exp
    log = tl.log
    log2 = tl.log2


if not hasattr(tl, 'gather'):

    @triton.jit
    def gather(src, index, axis, _builder=None):
        # This is a fallback implementation when tl.gather is not supported
        # In order to pass triton compiler, there is no actual gather operation
        return src
else:
    gather = tl.gather
