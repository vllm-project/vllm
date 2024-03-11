# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py

import dropout_layer_norm
import torch
from torch.nn import init


def maybe_align(x, alignment_in_bytes=16):
    """Assume that x already has last dim divisible by alignment_in_bytes"""
    # TD [2023-07-04] I'm not 100% sure that clone will align the memory
    # https://discuss.pytorch.org/t/how-to-ensure-that-tensor-data-ptr-is-aligned-to-16-bytes/183440
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


def _dropout_add_layer_norm_forward(
    x0,
    residual,
    gamma,
    beta,
    rowscale,
    colscale,
    dropout_p,
    epsilon,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat,
        residualmat,
        gamma,
        beta,
        rowscale,
        colscale,
        None,
        None,
        dropout_p,
        epsilon,
        1.0,
        0,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and residual is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(
    dz,
    dx,
    x,
    x0,
    dmask,
    mu,
    rsigma,
    gamma,
    rowscale,
    colscale,
    dropout_p,
    has_residual,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, "x0 is required to compute the gradient of colscale"
    dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat,
        dxmat,
        xmat,
        x0mat,
        dmask,
        mu,
        rsigma,
        gamma,
        rowscale,
        colscale,
        None,
        None,
        dropout_p,
        1.0,
        0,
        has_residual,
        is_rms_norm,
    )
    # dresidualmat is None if not has_residual
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_subset_forward(
    x0,
    residual,
    gamma,
    beta,
    colscale,
    x0_subset,
    out_subset,
    dropout_p,
    epsilon,
    rowscale_const,
    out_numrows,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat,
        residualmat,
        gamma,
        beta,
        None,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    # dmask is None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and residual is None and residual_dtype != input_dtype
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_subset_backward(
    dz,
    dx,
    x,
    x0,
    dmask,
    mu,
    rsigma,
    gamma,
    colscale,
    x0_subset,
    out_subset,
    dropout_p,
    rowscale_const,
    x0_numrows,
    has_residual,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    x0 must not be None if we have colscale.
    """
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(-1, hidden_size)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    if colscale is not None:
        assert x0 is not None, "x0 is required to compute the gradient of colscale"
    dx0mat, dresidualmat, dgamma, dbeta, _, _, *rest = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat,
        dxmat,
        xmat,
        x0mat,
        dmask,
        mu,
        rsigma,
        gamma,
        None,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        rowscale_const,
        x0_numrows,
        has_residual,
        is_rms_norm,
    )
    # dresidualmat is None if not has_residual
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_parallel_residual_forward(
    x0,
    x1,
    residual,
    gamma0,
    beta0,
    gamma1,
    beta1,
    dropout_p,
    epsilon,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes"""
    hidden_size = gamma0.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size)) if x1 is not None else None
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    (
        z0mat,
        z1mat,
        xmat,
        dmask0,
        dmask1,
        mu,
        rsigma,
    ) = dropout_layer_norm.dropout_add_ln_parallel_residual_fwd(
        x0mat,
        x1mat,
        residualmat,
        gamma0,
        beta0,
        gamma1,
        beta1,
        dropout_p,
        epsilon,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    # dmask0 and dmask1 are None if dropout_p == 0.0
    # xmat is None if dropout_p == 0.0 and residual is None and residual_dtype != input_dtype
    return z0mat, z1mat, xmat if xmat is not None else x0mat, dmask0, dmask1, mu, rsigma


def _dropout_add_layer_norm_parallel_residual_backward(
    dz0,
    dz1,
    dx,
    x,
    dmask0,
    dmask1,
    mu,
    rsigma,
    gamma0,
    gamma1,
    dropout_p,
    has_x1,
    has_residual,
    is_rms_norm=False,
):
    """Assume that arguments are contiguous and aligned to 16 bytes
    dx == None means that it was a post-norm architecture
    (x = drop(x0) + residual was not returned in the fwd).
    """
    hidden_size = gamma0.numel()
    xmat = x.view((-1, hidden_size))
    dz0mat = dz0.view(xmat.shape)
    dz1mat = dz1.view(xmat.shape) if dz1 is not None else None
    dxmat = dx.view(xmat.shape) if dx is not None else None
    (
        dx0mat,
        dx1mat,
        dresidualmat,
        dgamma0,
        dbeta0,
        dgamma1,
        dbeta1,
        *rest,
    ) = dropout_layer_norm.dropout_add_ln_parallel_residual_bwd(
        dz0mat,
        dz1mat,
        dxmat,
        xmat,
        dmask0,
        dmask1,
        mu,
        rsigma,
        gamma0,
        gamma1,
        dropout_p,
        has_x1,
        has_residual,
        is_rms_norm,
    )
    # dresidualmat is None if not has_residual
    return dx0mat, dx1mat, dresidualmat, dgamma0, dbeta0, dgamma1, dbeta1


class DropoutAddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        residual,
        gamma,
        beta,
        rowscale,
        colscale,
        dropout_p,
        epsilon,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        rowscale = maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
        colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0,
            residual,
            gamma,
            beta,
            rowscale,
            colscale,
            dropout_p,
            epsilon,
            residual_in_fp32,
            is_rms_norm,
        )
        # Only need to save x0 if we need to compute gradient wrt colscale
        x0_saved = x0 if colscale is not None else None
        ctx.save_for_backward(
            xmat.view(x0.shape), x0_saved, dmask, gamma, mu, rsigma, rowscale, colscale
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        if not return_dmask:
            return (
                zmat.view(x0.shape) if not prenorm else (zmat.view(x0.shape), xmat.view(x0.shape))
            )
        else:
            dmask = (
                dmask.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask)
            return (
                (zmat.view(x0.shape), dmask)
                if not prenorm
                else (zmat.view(x0.shape), xmat.view(x0.shape), dmask)
            )

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = maybe_align(dz.contiguous(), 16)  # this happens!
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
        # x0 is None if colscale is None
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(
            dz,
            dx,
            x,
            x0,
            dmask,
            mu,
            rsigma,
            gamma,
            rowscale,
            colscale,
            dropout_p,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(x.shape)
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (
            dx0,
            dresidual,
            dgamma,
            dbeta if ctx.has_beta else None,
            None,
            dcolscale,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DropoutAddLayerNormSubsetFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        residual,
        gamma,
        beta,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        colscale = maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_subset_forward(
            x0,
            residual,
            gamma,
            beta,
            colscale,
            x0_subset,
            out_subset,
            dropout_p,
            epsilon,
            rowscale_const,
            out_numrows,
            residual_in_fp32,
            is_rms_norm,
        )
        # Only need to save x0 if we need to compute gradient wrt colscale
        x0_saved = x0 if colscale is not None else None
        x_shape = (-1, *x0.shape[1:])
        ctx.save_for_backward(
            xmat.view(x_shape), x0_saved, dmask, gamma, mu, rsigma, colscale, x0_subset, out_subset
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.rowscale_const = rowscale_const
        ctx.x0_numrows = x0.shape[:-1].numel()
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        z_shape = (-1, *x0.shape[1:])
        if not return_dmask:
            return zmat.view(z_shape) if not prenorm else (zmat.view(z_shape), xmat.view(x0.shape))
        else:
            z = zmat.view(z_shape)
            dmask = (
                dmask.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask)
            return (z, dmask) if not prenorm else (z, xmat.view(x_shape), dmask)

    @staticmethod
    def backward(ctx, dz, *args):
        # assert dz.is_contiguous()
        dz = maybe_align(dz.contiguous(), 16)  # this happens!
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, colscale, x0_subset, out_subset = ctx.saved_tensors
        # x0 is None if colscale is None
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_subset_backward(
            dz,
            dx,
            x,
            x0,
            dmask,
            mu,
            rsigma,
            gamma,
            colscale,
            x0_subset,
            out_subset,
            dropout_p,
            ctx.rowscale_const,
            ctx.x0_numrows,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(-1, *x.shape[1:])
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (
            dx0,
            dresidual,
            dgamma,
            dbeta if ctx.has_beta else None,
            dcolscale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DropoutAddLayerNormParallelResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        x1,
        residual,
        gamma0,
        beta0,
        gamma1,
        beta1,
        dropout_p,
        epsilon,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        x1 = maybe_align(x1.contiguous(), 16) if x1 is not None else None
        residual = maybe_align(residual.contiguous(), 16) if residual is not None else None
        gamma0 = maybe_align(gamma0.contiguous(), 16)
        beta0 = maybe_align(beta0.contiguous(), 16) if beta0 is not None else None
        gamma1 = maybe_align(gamma1.contiguous(), 16) if gamma1 is not None else None
        beta1 = maybe_align(beta1.contiguous(), 16) if beta1 is not None else None
        (
            z0mat,
            z1mat,
            xmat,
            dmask0,
            dmask1,
            mu,
            rsigma,
        ) = _dropout_add_layer_norm_parallel_residual_forward(
            x0,
            x1,
            residual,
            gamma0,
            beta0,
            gamma1,
            beta1,
            dropout_p,
            epsilon,
            residual_in_fp32,
            is_rms_norm,
        )
        ctx.save_for_backward(xmat.view(x0.shape), dmask0, dmask1, gamma0, gamma1, mu, rsigma)
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_x1 = x1 is not None
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta0 is not None
        z = (z0mat.view(x0.shape), z1mat.view(x0.shape) if z1mat is not None else None)
        if not return_dmask:
            return z if not prenorm else (*z, xmat.view(x0.shape))
        else:
            dmask0 = (
                dmask0.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            dmask1 = (
                dmask1.view(x0.shape)
                if dropout_p > 0.0 and x1 is not None
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask0)
            ctx.mark_non_differentiable(dmask1)
            return (
                (*z, dmask0, dmask1) if not prenorm else (*z, xmat.view(x0.shape), dmask0, dmask1)
            )

    @staticmethod
    def backward(ctx, dz0, dz1, *args):
        dz0 = maybe_align(dz0.contiguous(), 16)  # this happens!
        dz1 = maybe_align(dz1.contiguous(), 16) if dz1 is not None else None
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, dmask0, dmask1, gamma0, gamma1, mu, rsigma = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_x1 = ctx.has_x1
        has_residual = ctx.has_residual
        (
            dx0mat,
            dx1mat,
            dresidualmat,
            dgamma0,
            dbeta0,
            dgamma1,
            dbeta1,
        ) = _dropout_add_layer_norm_parallel_residual_backward(
            dz0,
            dz1,
            dx,
            x,
            dmask0,
            dmask1,
            mu,
            rsigma,
            gamma0,
            gamma1,
            dropout_p,
            has_x1,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        return (
            dx0,
            dx1,
            dresidual,
            dgamma0,
            dbeta0 if ctx.has_beta else None,
            dgamma1,
            dbeta1 if ctx.has_beta else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def layer_norm(x, weight, bias, epsilon):
    return DropoutAddLayerNormFn.apply(x, None, weight, bias, None, None, 0.0, epsilon, False)


def dropout_add_layer_norm(
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
        False,
        return_dropout_mask,
    )


def dropout_add_layer_norm_subset(
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
        False,
        return_dropout_mask,
    )


def dropout_add_layer_norm_parallel_residual(
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
        False,
        return_dropout_mask,
    )


class DropoutAddLayerNorm(torch.nn.Module):
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
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, residual=None):
        return dropout_add_layer_norm(
            x0,
            residual,
            self.weight,
            self.bias,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )
