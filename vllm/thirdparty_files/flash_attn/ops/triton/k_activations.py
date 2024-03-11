# Adapted from https://github.com/facebookresearch/xformers/blob/main/xformers/triton/k_activations.py
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from typing import Optional

import triton
import triton.language as tl

_sqrt2pi = math.sqrt(2.0 / math.pi)
_sqrt1_2 = math.sqrt(1.0 / 2)
_gaussian_pdf_normalization = 1.0 / math.sqrt(2 * math.pi)


class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    GeLU = "gelu"
    GeLUApprox = "gelu_approx"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"


def get_triton_activation_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu,
            Activation.LeakyReLU: leaky_relu,
            Activation.GeLU: gelu,
            Activation.GeLUApprox: gelu_approx,
            Activation.SquaredReLU: squared_relu,
        }[activation]
        if activation
        else None
    )


def get_triton_activation_bwd_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.ReLU: relu_grad,
            Activation.LeakyReLU: leaky_relu_grad,
            Activation.GeLU: gelu_grad,
            Activation.GeLUApprox: gelu_approx_grad,
            Activation.SquaredReLU: squared_relu_grad,
        }[activation]
        if activation
        else None
    )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


# a Triton implementation of the most used activations
# See for instance http://arxiv.org/abs/1606.08415 for an overview

# ReLU
@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))


@triton.jit
def relu_grad(x):
    # ReLU is different from other activations
    # in that it does not require the input to retrospectively compute its gradient
    # here the input is the downstream gradient, and we return the upstream gradient directly
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one.to(x.dtype), zero.to(x.dtype))


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return (x_ * x_).to(x.dtype)


@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0, 2.0 * x, 0.0)


# Leaky ReLU
@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    scale = 0.01 + 0.0
    scale = scale.to(x.dtype)
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1

    min_grad = min_grad.to(x.dtype)
    max_grad = max_grad.to(x.dtype)

    return tl.where(x >= 0, max_grad, min_grad)


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))


@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))
    pdf = tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization
    return cdf + x * pdf


@triton.jit
def gelu_approx(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def gelu_approx_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
