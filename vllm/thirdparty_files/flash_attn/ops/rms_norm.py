# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py

import torch
from torch.nn import init

from flash_attn.ops.layer_norm import (
    DropoutAddLayerNormFn,
    DropoutAddLayerNormParallelResidualFn,
    DropoutAddLayerNormSubsetFn,
)


def rms_norm(x, weight, epsilon):
    return DropoutAddLayerNormFn.apply(
        x, None, weight, None, None, None, 0.0, epsilon, False, False, True
    )


def dropout_add_rms_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormFn.apply(
        x0,
        residual,
        weight,
        bias,
        rowscale,
        layerscale,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        True,
        return_dropout_mask,
    )


def dropout_add_rms_norm_subset(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    layerscale=None,
    x0_subset=None,
    out_subset=None,
    rowscale_const=1.0,
    out_numrows=0,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormSubsetFn.apply(
        x0,
        residual,
        weight,
        bias,
        layerscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        residual_in_fp32,
        prenorm,
        True,
        return_dropout_mask,
    )


def dropout_add_rms_norm_parallel_residual(
    x0,
    x1,
    residual,
    weight0,
    bias0,
    weight1,
    bias1,
    dropout_p,
    epsilon,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    """residual_in_fp32 only has an effect if residual is None.
    Otherwise residual dtype is residual.dtype.
    """
    return DropoutAddLayerNormParallelResidualFn.apply(
        x0,
        x1,
        residual,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        True,
        return_dropout_mask,
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class DropoutAddRMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x0, residual=None):
        return dropout_add_rms_norm(
            x0,
            residual,
            self.weight,
            None,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )
