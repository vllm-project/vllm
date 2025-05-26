# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Code copied from Microsoft/MoE by Jacob Platin (jacobplatin@microsoft.com)
# but implemented by the Phi-Speech team
#!/usr/bin/env python3
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BlockBase(nn.Module):
    """Block abstract module"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


def get_activation(name="relu"):
    """Select an activation function by name

    Args:
        name: str
            activation function name,
            one of ["relu", "gelu", "swish", "sigmoid"],
            default "relu".
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "swish":
        return Swish()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    return nn.Identity()


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. 
        It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for 
        chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long(
    )  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(
        chunk_start_idx,
        (1, 0))  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0,
                             x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) &
           (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    # boundary = end_pad[idx]  # boundary size: [x_len]
    seq_range_expand = (torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)
                        )  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right


class Swish(nn.Module):
    """Implement Swish activation module.
    From https://arxiv.org/pdf/2005.03191.pdf

    """

    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Apply Swish function

        Args:
            x: torch.Tensor
                Input.
        """
        return x * self.act_fn(x)


class GLU(nn.Module):
    """Implement Gated Linear Unit (GLU) module"""

    def __init__(self, dim: int = -1, act_name: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.act_name = act_name.lower()

        if self.act_name == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif self.act_name == "gelu":
            self.act_fn = nn.GELU()
        elif self.act_name == "swish":
            self.act_fn = Swish()
        elif self.act_name == "sigmoid":
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """GLU forward
        Apply Swish function on the first half of input matrices
        with sigmoid of the second half.

        Args:
            x: torch.Tensor
                Input.

        """
        half_x, gate = x.chunk(2, dim=self.dim)
        return half_x * self.act_fn(gate)


# TODO: Abdel, this can be improved using GLU module
class GLUPointWiseConv(nn.Module):
    """GLUPointWiseConv module
    used for conformer architecture,
    for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        output_dim: int
            output channel size.
        kernel_size: int
            kernel size
        glu_type: str, optional
            activation function one of
             ["sigmoid", "relu", "gelu"]
              default "sigmoid".
        bias_in_glu: bool, optional
            use addtive bias in glu
        causal: bool, optional
            if set to True, padding is set to the half of
             kernel size, ie, convolution can't see future frames.
              default False.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        glu_type="sigmoid",
        bias_in_glu=True,
        causal=False,
    ):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim
        self.bias_in_glu = bias_in_glu
        if causal:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1),
            )
        else:
            self.ext_pw_conv_1d = nn.Conv1d(
                input_dim,
                output_dim * 2,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            )

        if glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, output_dim, 1))
            self.b2 = nn.Parameter(torch.zeros(1, output_dim, 1))

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                input tensor
        """
        # to be consistent with GLULinear, we assume the input always has the
        # #channel (#dim) in the last dimension of the tensor, so need to
        # switch the dimension first for 1D-Conv case
        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.glu_type == "bilinear":
            if self.bias_in_glu:
                x = (x[:, 0:self.output_dim, :] + self.b1) * (
                    x[:, self.output_dim:self.output_dim * 2, :] + self.b2)
            else:
                x = (x[:, 0:self.output_dim, :]) * (
                    x[:, self.output_dim:self.output_dim * 2, :])
        else:
            if self.bias_in_glu:
                x = (x[:, 0:self.output_dim, :] + self.b1) * self.glu_act(
                    x[:, self.output_dim:self.output_dim * 2, :] + self.b2)
            else:
                x = (x[:, 0:self.output_dim, :]) * self.glu_act(
                    x[:, self.output_dim:self.output_dim * 2, :])

        x = x.permute([0, 2, 1])
        return x


class DepthWiseSeperableConv1d(nn.Module):
    """DepthWiseSeperableConv1d module used in Convnet module
    for the conformer, for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of 
             depthwise_seperable_out_channel will be used as a channel_out
             of the second conv1d layer.
             otherwise, it equal to 0, the second conv1d layer is skipped.
        kernel_size: int
            kernel_size
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
            will be used to compute the hidden channels of the Conv1D.
        padding: int, optional
            padding for the conv1d,
             default: 0.

    """

    def __init__(
        self,
        input_dim,
        depthwise_seperable_out_channel,
        kernel_size,
        depthwise_multiplier,
        padding=0,
    ):
        super().__init__()

        self.dw_conv = nn.Conv1d(
            input_dim,
            input_dim * depthwise_multiplier,
            kernel_size,
            1,
            padding=padding,
            groups=input_dim,
        )

        if depthwise_seperable_out_channel != 0:
            self.pw_conv = nn.Conv1d(
                input_dim * depthwise_multiplier,
                depthwise_seperable_out_channel,
                1,
                1,
                0,
            )
        else:
            self.pw_conv = nn.Identity()
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel

    def forward(self, x):
        """

        Args:
            x: torch.Tensor
                input tensor
        """
        x = self.dw_conv(x)
        if self.depthwise_seperable_out_channel != 0:
            x = self.pw_conv(x)
        return x


class ConvModule(nn.Module):
    """ConvModule Module for the conformer block.
    for more details see:
    https://arxiv.org/pdf/2005.08100v1.pdf

    Args:
        input_dim: int
            input channel size.
        ext_pw_out_channel: int
            if > 0, ext_pw_out_channel is a dim channel size
             for the last pointwise conv after swish activation.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of 
             depthwise_seperable_out_channel
             will be used as a channel_out of the second conv1d layer.
             otherwise, it equal to 0, the second conv1d layer is skipped.
        ext_pw_kernel_size: int
            kernel size of the conv pointwise of the conformer.
        kernel_size: int
            kernel size.
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
             will be used to compute the hidden channels of the Conv1D.
        dropout_rate: float
            dropout rate.
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation.
            default False
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
        chunk_size: int, optional
            chunk size for cnn. default 18
        activation: str, optional
            activation function used in ConvModule,
            default: "relu".
        glu_type: str, optional
            activation function used for the glu,
            default: "sigmoid".
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU.
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        export: bool, optional,
            if set to True, padding is equal to 0.  This is for inference,
             or onnx export.  Typically this is set by the export program or
             the decoder program, and it isn't present in your config file.
             default False
    """

    def __init__(
        self,
        input_dim,
        ext_pw_out_channel,
        depthwise_seperable_out_channel,
        ext_pw_kernel_size,
        kernel_size,
        depthwise_multiplier,
        dropout_rate,
        causal=False,
        batch_norm=False,
        chunk_se=0,
        chunk_size=18,
        activation="relu",
        glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        export=False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_dim = input_dim
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.glu_type = glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.causal = causal

        self._add_ext_pw_layer()

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size

        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(input_dim)

        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.export = export

        if causal:
            padding = 0 if export else kernel_size - 1
        else:
            padding = (kernel_size - 1) // 2

        self.dw_sep_conv_1d = DepthWiseSeperableConv1d(
            input_dim,
            depthwise_seperable_out_channel,
            kernel_size,
            depthwise_multiplier,
            padding=padding,
        )

        if depthwise_seperable_out_channel != 0:
            if input_dim != depthwise_seperable_out_channel:
                self.ln2 = nn.Linear(depthwise_seperable_out_channel,
                                     input_dim)
        else:
            if depthwise_multiplier != 1:
                self.ln2 = nn.Linear(input_dim * depthwise_multiplier,
                                     input_dim)

    def _add_ext_pw_layer(self):
        """
        This function is an extension of __init__ function
        and dedicated to the convolution module creation
        of the conformer.
        """
        self.ln1 = self.glu = self.bn_layer = self.ext_pw_conv_1d = (
            nn.Identity())  # jit hacks.
        self.squeeze_excitation = nn.Identity()  # jit.
        self.apply_ln1 = self.fix_len1 = False  # jit.

        if self.ext_pw_out_channel != 0:
            if self.causal:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1),
                )
                if self.ext_pw_kernel_size > 1:
                    self.fix_len1 = True
                else:
                    self.fix_len1 = False
            else:
                self.ext_pw_conv_1d = nn.Conv1d(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    1,
                    padding=(self.ext_pw_kernel_size - 1) // 2,
                )
                self.fix_len1 = False

            if self.linear_glu_in_convm:
                self.glu = GLULinear(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.glu_type,
                    self.bias_in_glu,
                )
            else:
                self.glu = GLUPointWiseConv(
                    self.input_dim,
                    self.ext_pw_out_channel,
                    self.ext_pw_kernel_size,
                    self.glu_type,
                    self.bias_in_glu,
                    self.causal,
                )

            if self.input_dim != self.ext_pw_out_channel:
                self.apply_ln1 = True
                self.ln1 = nn.Linear(self.ext_pw_out_channel, self.input_dim)
            else:
                self.apply_ln1 = False
        else:
            self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
            self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x):
        """ConvModule Forward.

        Args:
            x: torch.Tensor
                input tensor.
        """
        x = self.layer_norm(x)

        if self.ext_pw_out_channel != 0:
            x = self.glu(x)
            if self.causal and self.ext_pw_kernel_size > 1:
                x = x[:, :-(self.ext_pw_kernel_size - 1), :]
            if self.apply_ln1:
                x = self.ln1(x)
        else:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1

        x = x.permute([0, 2, 1])

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, :-(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        if self.ext_pw_out_channel != 0:
            x = self.ext_pw_conv_1d(x)
            if self.fix_len1:
                x = x[:, :, :-(self.ext_pw_kernel_size - 1)]

            if self.apply_ln1:
                x = x.permute([0, 2, 1])
                x = self.ln1(x)
                x = x.permute([0, 2, 1])

            x = x.permute([0, 2, 1])
        else:
            x = x.unsqueeze(1).permute([0, 1, 3, 2])
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
            x = x.squeeze(1)

        x = self.dropout(x)
        return x


class GLULinear(nn.Module):
    """Linear + GLU module

    Args:
        input_dim: int
            input size
        output_dim: int
            output size.
        glu_type:
            activation function name used in glu module.
            default "sigmoid" (swish function).
        bias_in_glu: bool, optional
            If True, the addtive bias is added. Default False.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        glu_type="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2, bias_in_glu)
        self.glu_act = GLU(-1, glu_type)

    def forward(self, x):
        """GLULinear forward

        Args:
            x: torch.Tensor
                inpute tensor.
        """
        x = self.linear(x)
        return self.glu_act(x)


class FeedForward(nn.Module):
    """FeedForward Module.
    For more details see Conformer paper:
        https://arxiv.org/pdf/2005.08100.pdf

    Args:
        d_model: int
            input size.
        d_inner: int
            output size.
        dropout_rate: float,
            dropout rate.
        activation: str,
            activation function name,
            one of ["relu", "swish", "sigmoid"],
            sigmoid activation is only used with "glu_in_fnn=True",
            default "sigmoid".
        bias_in_glu: bool, optional
    """

    def __init__(
        self,
        d_model,
        d_inner,
        dropout_rate,
        activation="sigmoid",
        bias_in_glu=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        module = GLULinear(d_model, d_inner, activation, bias_in_glu)
        self.net = nn.Sequential(
            module,
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """FeedForward forward function.

        Args:
            x: torch.Tensor
                input tensor.
        """
        out = self.net(self.layer_norm(x))

        return out


#### positional encoding starts here
def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Perform pre-hook in load_state_dict for backward compatibility.

    Note:
        We saved self.pe until v.0.5.2 but we have omitted it later.
        Therefore, we remove the item "pe" from `state_dict` for backward 
        compatibility.

    """
    k = prefix + "pe"
    if k in state_dict:
        state_dict.pop(k)


class T5RelativeAttentionLogitBias(nn.Module):
    """
    This module implements the relative position bias described in Section 
    2.1 of the T5 paper: https://arxiv.org/pdf/1910.10683.pdf

    The Huggingface implementation is used as a reference
    https://github.com/huggingface/transformers/blob/v4.30.0/src/
    transformers/models/t5/modeling_t5.py#L435

    Modifies attention as Q*K^T + B, where B is a learned scalar bias based
    on relative position of the query and key. It is HxNxN, where H is the 
    number of heads, N is the sequence length.

    I've made these modifications to the original T5 bias:
    - Skipping of the bucketing step. Original T5 bias converted rel 
      position distances into logarithmically increasing buckets. This is 
      supposed to help with length generalization.
    - I just directly use rel position index as bias values, as we don't 
      need length generalization (40s max is good enough for ASR encoder), 
      and it keeps ONNX export simple.
    - I've also extended it so that biases can be asymmetric, the default 
      implementation treats L->R and R->L the same. Asymmetric was found to 
      yield better results in my experiments.

    Args:
        num_heads: int
            Number of attention heads
        num_buckets: int
            Number of buckets to use for relative attention bias. This is the
            size of the learnable bias parameter. Bucketing is not yet 
            supported, so this defaults to -1 which means no bucketing is
            used (max_distance determines size of bias param).
        max_distance: int
            Maximum distance to use for relative attention bias. With 
            num_buckets=-1, this directly controls the max size of the bias 
            parameter. When num_buckets > 0 is supported, this will control 
            the maximum distance for logarithmic bucketing after which all 
            positions are in the same bucket.
        symmetric: bool
            Whether to use symmetric or asymmetric biases. symmetric=False uses
            2x number of bias params to distinguish L->R from R->L. This was 
            found to be better for the encoder.
    """

    def __init__(self,
                 num_heads,
                 num_buckets=-1,
                 max_distance=1000,
                 symmetric=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.symmetric = symmetric
        self._skip_bucketing = self.num_buckets < 0
        if self._skip_bucketing:
            self.num_buckets = max_distance
        else:
            raise NotImplementedError(
                "T5 attention bias with bucketed positions is not yet tested")
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        maxpos = x.size(1)
        context_position = torch.arange(maxpos,
                                        device=x.device,
                                        dtype=torch.long)[:, None]
        memory_position = torch.arange(maxpos,
                                       device=x.device,
                                       dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX
        # export
        relative_position = relative_position.masked_fill(
            relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1)

        # mapping from relative position to index in the bias parameter
        if self._skip_bucketing:
            bias_idx = relative_position
        else:
            bias_idx = self._bucket_relative_position(relative_position)
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2

        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(
            0)  # [1, H, L, L]

        return t5_rel_att_bias

    def _bucket_relative_position(self, relative_position):
        # this is a placeholder (isn't tested, likely buggy) using HuggingFace
        # implem as a reference this also needs to be extended to support
        # asymmetric +/- ve positions
        relative_buckets = 0
        if not self.causal:
            self.num_buckets //= 2
            relative_buckets += (relative_position > 0).to(
                torch.long) * self.num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in
        # positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            math.log(self.max_distance / max_exact) *
            (self.num_buckets - max_exact)).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, self.num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position,
                                        relative_position_if_large)
        return relative_buckets


class AbsolutePositionalEncoding(nn.Module):
    """Absolute Positional encoding module.
    This module implement Absolute sinusoidal positional encoding
    from: https://arxiv.org/pdf/1706.03762.pdf

    Args:
        d_model: int
            Input embedding size.
        dropout_rate: float
            dropout rate
        max_len: int, optional
            Maximum input length sequence, Default 5000

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings.

        Args:
            x: torch.Tensor
        """
        if self.pe is not None and self.pe.size(1) >= x.size(1):
            if self.pe.dtype != x.dtype or self.pe.device != x.device:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
            return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x: torch.Tensor
                Input tensor. shape is (batch, time, ...)

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


#### forward embedding layers starts here
class MeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.

    Will subtract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        input_size: int
            layer input size.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.global_mean = nn.Parameter(torch.zeros(input_size))
        self.global_invstd = nn.Parameter(torch.ones(input_size))

    def forward(self, input_: Tensor) -> Tensor:
        """MeanVarianceNormLayer Forward

        Args:
            input_: torch.Tensor
                input tensor.
        """
        return (input_ - self.global_mean) * self.global_invstd


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to
    locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a 
    causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as 
    left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible
    on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should 
    be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError(
                    "No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (isinstance(padding, list) and len(padding) == 2
                  and padding[0] + padding[1] == kernel_size - 1):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, :-self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1):]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would
    have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be 
    set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError(
                "Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,
    ):
        x = F.pad(
            x,
            pad=(self._left_padding, self._right_padding, 0, 0),
        )
        x = super().forward(x)
        return x


class NemoConvSubsampling(torch.nn.Module):
    """Convlutional subsampling module, taken from NeMo ASR
    (https://github.com/NVIDIA/NeMo/blob/b367413645d5c72db3c2c96e46e95a
    34501479cf/nemo/collections/asr/parts/submodules/subsampling.py)

    Striding Subsampling: "Speech-Transformer: A No-Recurrence 
    Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong 
    et al. (https://ieeexplore.ieee.org/document/8462506)


    Compared with the EncoderConv2D (`input_layer: custom`), this is a 
    much simplified approach, and uses no LayerNorm and far fewer Conv2Ds.
    Moreover, depthwise convolutions are used to reduce FLOPs, but the first
      layer is kept as a regular convolution so as not to degrade accuracy.

    `Striding` and `dw_striding` are the same except that the latter uses 
    depthwise convolutions after the first layer, whereas the former does not.

    Args:
        subsampling_factor (int): Time reduction factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        subsampling (str): The subsampling technique, choose from
            {"striding", "dw-striding", "striding_conv1d", 
            "dw_striding_conv1d"}
        conv_channels (int): Number of channels for the convolution layers, 
                            default is 256.
        subsampling_conv_chunking_factor (int): Input chunking factor which 
            can be -1 (no chunking) 1 (auto) or a power of 2. Default is 1
        activation (Module): activation function, default is nn.ReLU()
        is_causal (bool): whether to use causal Conv1/2D, where each step will
            have limited access to locations on its right or left
    """

    def __init__(
            self,
            feat_in,
            feat_out,
            subsampling_factor=4,
            subsampling="dw_striding",
            conv_channels=256,
            subsampling_conv_chunking_factor=1,
            activation=nn.ReLU(),  # noqa: B008
            is_causal=False,
    ):
        super().__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.is_causal = is_causal
        self.subsampling_causal_cond = subsampling in (
            "dw_striding",
            "striding",
            "striding_conv1d",
        )

        if (subsampling_conv_chunking_factor != -1
                and subsampling_conv_chunking_factor != 1
                and subsampling_conv_chunking_factor % 2 != 0):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a "\
                    "power of 2"
            )
        self.subsampling_conv_chunking_factor = \
            subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        if subsampling == "dw_striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            # Layer 1
            if self.is_causal:
                layers.append(
                    CausalConv2D(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=None,
                    ))
            else:
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                    ))
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                            groups=in_channels,
                        ))
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                            groups=in_channels,
                        ))

                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ))
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding":
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        ))
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        ))
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            if self.is_causal:
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = subsampling_factor + 1
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv1D(
                            in_channels=in_channels,
                            out_channels=(feat_out if self._sampling_num == i +
                                          1 else conv_channels),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=None,
                        ))
                else:
                    layers.append(
                        torch.nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=(feat_out if self._sampling_num == i +
                                          1 else conv_channels),
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        ))
                layers.append(activation)
                in_channels = conv_channels

        elif subsampling == "dw_striding_conv1d":
            in_channels = feat_in

            self._stride = 2
            self._kernel_size = 5
            self._ceil_mode = False

            self._left_padding = (self._kernel_size - 1) // 2
            self._right_padding = (self._kernel_size - 1) // 2

            # Layer 1
            layers.extend([
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                ),
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=(feat_out if self._sampling_num == 1 else
                                  conv_channels),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                ),
            ])
            in_channels = conv_channels
            layers.append(activation)

            for i in range(self._sampling_num - 1):
                layers.extend([
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._left_padding,
                        groups=in_channels,
                    ),
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=(feat_out if self._sampling_num == i +
                                      2 else conv_channels),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                    ),
                ])
                layers.append(activation)
                in_channels = conv_channels

        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        if subsampling in ["dw_striding", "striding"]:
            in_length = torch.tensor(feat_in, dtype=torch.float)
            out_length = calc_length(
                lengths=in_length,
                all_paddings=self._left_padding + self._right_padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
                repeat_num=self._sampling_num,
            )
            self.out = torch.nn.Linear(conv_channels * int(out_length),
                                       feat_out)
            self.conv2d_subsampling = True
        elif subsampling in ["striding_conv1d", "dw_striding_conv1d"]:
            self.out = None
            self.conv2d_subsampling = False
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_factor]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_factor + 1]

    def forward(self, x, mask):
        """
        Forward method for NeMo subsampling.

        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            x_mask: torch.Tensor
                input mask

        Returns:
            x: torch.Tensor
                Resulting tensor from subsampling (B, T // 
                time_reduction_factor, feat_out)
            pad_mask: torch.Tensor
                tensor of padded hidden state sequences (B, 1, T // 
                time_reduction_factor)
        """
        x = x.unsqueeze(1) if self.conv2d_subsampling else x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if (self.subsampling_conv_chunking_factor != -1
                and self.conv2d_subsampling):
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only
                # if needed.
                # avoiding a bug / feature limiting indexing of tensors
                # to 2**31.
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = (2**31 / self._conv_channels * self._stride *
                          self._stride)
                need_to_split = torch.numel(x) > x_ceil
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                if not success:  # if unable to split by batch, try by channel
                    if self._subsampling == "dw_striding":
                        x = self.conv_split_by_channel(x)
                    else:
                        x = self.conv(x)  # try anyway
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        if self.is_causal and self.subsampling_causal_cond:
            feature_lens_remainder = feature_lens % self.subsampling_factor
            padding_length[feature_lens_remainder != 1] += 1
        pad_mask = torch.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1) < padding_length.unsqueeze(1)
        return x, pad_mask.unsqueeze(1)

    def reset_parameters(self):
        # initialize weights
        if self._subsampling == "dw_striding":
            with torch.no_grad():
                # init conv
                scale = 1.0 / self._kernel_size
                dw_max = (self._kernel_size**2)**-0.5
                pw_max = self._conv_channels**-0.5

                torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
                torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

                for idx in range(2, len(self.conv), 3):
                    torch.nn.init.uniform_(self.conv[idx].weight, -dw_max,
                                           dw_max)
                    torch.nn.init.uniform_(self.conv[idx].bias, -dw_max,
                                           dw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max,
                                           pw_max)
                    torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max,
                                           pw_max)

                # init fc (80 * 64 = 5120 from https://github.com/kssteven418/
                # Squeezeformer/blob/13c97d6cf92f2844d2cb3142b4c5bfa9ad1a8951/
                # src/models/conformer_encoder.py#L487
                fc_scale = (self._feat_out * self._feat_in /
                            self._sampling_num)**-0.5
                torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
                torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x):
        """Tries to split input by batch, run conv and concat results"""
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2**p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        return (
            torch.cat([
                self.conv(chunk)
                for chunk in torch.split(x, new_batch_size, 0)
            ]),
            True,
        )

    def conv_split_by_channel(self, x):
        """For dw convs, tries to split input by time, run conv and concat 
        results"""
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors
                # to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2**31, 2))
                cf = 2**p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c,
                                          x)  # conv2D, depthwise

            # splitting pointwise convs by time
            x = torch.cat(
                [
                    self.conv[i * 3 + 3](chunk)
                    for chunk in torch.split(x, new_t, 2)
                ],
                2,
            )  # conv2D, pointwise
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]

            if self.is_causal:
                chunk = nn.functional.pad(
                    chunk,
                    pad=(
                        self._kernel_size - 1,
                        self._stride - 1,
                        self._kernel_size - 1,
                        self._stride - 1,
                    ),
                )
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind:ind + step, :, :, :],
                    bias=conv.bias[ind:ind + step],
                    stride=self._stride,
                    padding=0,
                    groups=step,
                )
            else:
                ch_out = nn.functional.conv2d(
                    chunk,
                    conv.weight[ind:ind + step, :, :, :],
                    bias=conv.bias[ind:ind + step],
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=step,
                )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def change_subsampling_conv_chunking_factor(
            self, subsampling_conv_chunking_factor: int):
        if (subsampling_conv_chunking_factor != -1
                and subsampling_conv_chunking_factor != 1
                and subsampling_conv_chunking_factor % 2 != 0):
            raise ValueError(
                "subsampling_conv_chunking_factor should be -1, 1, or a "\
                    "power of 2"
            )
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor


def calc_length(lengths,
                all_paddings,
                kernel_size,
                stride,
                ceil_mode,
                repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or
      max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = (torch.div(lengths.to(dtype=torch.float) + add_pad, stride) +
                   one)
        lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
    return lengths.to(dtype=torch.int)


####  multihead attention starts here
class AttModule(nn.Module):
    """Attention abstraction module"""

    def __init__(self):
        super().__init__()
        self.export_mode = False

    def set_export(self, mode=True):
        """set the export mode"""
        self.export_mode = mode

    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        att_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """AttModule forward

        Args:
            x: torch.Tensor
                input tensor.
            memory: torch.Tensor, optional
                memory tensor.
            pos_emb: torch.Tensor, optional
                positional encoder embedding.
            att_mask: torch.Tensor, optional
                attention mask tensor.
        """
        return x, memory, pos_emb, att_mask


class AttBlock(BlockBase, AttModule):
    """Attention Block module to support both Attention and Block module."""

    def memory_dims(self, max_len=False):
        """memory dimensions"""
        return (1, self.input_size)


def masked_softmax(
    scores,
    mask: Optional[Tensor],
):
    if mask is not None:
        mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
        scores = scores.masked_fill(mask, -torch.inf)
        attn = torch.softmax(scores, dim=-1).masked_fill(
            mask, 0.0)  # (batch, head, time1, time2)
    else:
        attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
    return attn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer with optional relative position embedding 
    and GLU.

    Args:
        n_head: int
            the number of heads.
        n_feat: int
            input size features.
        dropout_rate: float
            dropout rate.
        use_LN: bool
            apply layer norm or not
        dropout_at_output: bool
            whether to apply dropout at output
        attention_inner_dim: int, optional
            the attention dimension used in the class,
            it can be different from the input dimension n_feat.
            default: -1 (equal to n_feat).
        use_pt_scaled_dot_product_attention: bool, optional
            if set True, use pytorch scaled dot product attention in training.
            NOTE: this will NOT be used in ONNX decoding due to a lack of 
            support.  In that case, we use the original attention 
            implementation, which shows no regression.
            default: False.
        n_value: int, optional
            if set to values other than -1, use a different dimension for 
            value. With the default value (i.e. -1), it is backward compatible.
        group_size: int, optional. must divide `n_head`
            if group_size > 1:       GQA
            if group_size = 1:       MHA
            if group_size = n_head:  MQA
    """

    inv_sqrt_d_k: torch.jit.Final[float]
    h: torch.jit.Final[int]
    h_k: torch.jit.Final[int]
    g: torch.jit.Final[int]

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        attention_inner_dim=-1,
        glu_type="swish",
        bias_in_glu=True,
        use_pt_scaled_dot_product_attention=False,
        n_value=-1,
        group_size: int = 1,
    ):
        super().__init__()
        if n_value == -1:
            n_value = n_feat
        if attention_inner_dim == -1:
            attention_inner_dim = n_feat
        assert attention_inner_dim % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = attention_inner_dim // n_head
        self.inv_sqrt_d_k = 1.0 / math.sqrt(self.d_k)
        self.h = n_head
        assert n_head % group_size == 0, "group_size must divide n_head"
        self.g = group_size
        self.h_k = n_head // group_size

        self.linear_q = nn.Linear(n_feat, attention_inner_dim)
        self.linear_k = nn.Linear(n_feat, attention_inner_dim // group_size)
        self.linear_v = nn.Linear(n_value, attention_inner_dim // group_size)
        self.linear_out = nn.Linear(attention_inner_dim // group_size, n_value)

        self.attn = torch.jit.Attribute(None, Optional[Tensor])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_rate = dropout_rate
        self.use_pt_scaled_dot_product_attention = (
            use_pt_scaled_dot_product_attention)

        if use_pt_scaled_dot_product_attention and group_size > 1:
            raise ValueError("Cannot use PT Scaled Attention with GQA")

        # Torchscript eager quantization.  Note that these functions below are
        # NOOPs and have very little impact on performance unless quantization
        # is enabled.
        self.quant_q = torch.ao.quantization.QuantStub()
        self.quant_x = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.ffunc = torch.ao.nn.quantized.FloatFunctional()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_k: Tensor,
        pos_v: Tensor,
        mask: Optional[Tensor],
        relative_attention_bias: Optional[Tensor] = None,
    ):
        """Compute 'Scaled Dot Product Attention'.

        Args:
            query: torch.Tensor
                query tensor (batch, time1, size)
            key: torch.Tensor
                key tensor (batch, time2, size)
            value: torch.Tensor
                value tensor (batch, time1, size)
            pos_k: torch.Tensor
                key tensor used for relative positional embedding.
            pos_v: torch.Tensor
                value tensor used for relative positional embedding.
            mask: torch.Tensor
                mask tensor (batch, time1, time2)
            relative_attention_bias: torch.Tensor
                bias added to attention logits w.r.t. relative positions
                (1, n_head, time1, time2)
        """
        n_batch = query.size(0)

        q = self.linear_q(query).view(n_batch, -1, self.h,
                                      self.d_k)  # (b, t, d)
        k = self.linear_k(key).view(n_batch, -1, self.h_k,
                                    self.d_k)  # (b, t, d)
        v = self.linear_v(value).view(n_batch, -1, self.h_k, self.d_k)
        q = (q.transpose(1, 2) if self.use_pt_scaled_dot_product_attention
             and not torch.jit.is_scripting() else q.transpose(1, 2) *
             self.inv_sqrt_d_k)
        k = k.transpose(1, 2)  # (batch, head_k, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head_k, time2, d_k)

        if (self.use_pt_scaled_dot_product_attention
                and not torch.jit.is_scripting()):
            attn_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)
                if relative_attention_bias is not None:
                    attn_mask = mask + relative_attention_bias
                else:
                    attn_mask = mask
                if mask.dtype != q.dtype:
                    attn_mask = attn_mask.to(q.dtype)

            with torch.nn.attention.sdpa_kernel([
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            ]):
                x = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_rate,
                )
        else:
            if self.h != self.h_k:
                q = q.reshape(n_batch, self.g, self.h_k, -1, self.d_k)
                A = torch.einsum("b g h t d, b h s d -> b h t s", q, k)
            else:
                A = torch.matmul(q, k.transpose(-2, -1))
            if pos_k is not None:
                if self.h != self.h_k:
                    B = torch.einsum("b g h t d, t s d -> b h t s", q, pos_k)
                else:
                    reshape_q = (q.contiguous().view(n_batch * self.h, -1,
                                                     self.d_k).transpose(0, 1)
                                 )  # (t1,nh,dk)
                    B = torch.matmul(reshape_q,
                                     pos_k.transpose(-2,
                                                     -1))  # pos_k: (t1,dk,t2)
                    B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0),
                                               pos_k.size(1))
                scores = A + B
            else:
                scores = A

            if relative_attention_bias is not None:
                scores = scores + relative_attention_bias

            attn = masked_softmax(scores, mask)  # (batch, head, time1, time2)

            self.attn = attn

            p_attn = self.dropout(attn)
            x = torch.matmul(p_attn.to(v.dtype),
                             v)  # (batch, head, time1, d_k)
            if pos_v is not None:
                reshape_attn = (p_attn.contiguous().view(
                    n_batch * self.h, pos_v.size(0),
                    pos_v.size(1)).transpose(0, 1))  # (t1, bh, t2)

                attn_v = (torch.matmul(reshape_attn, pos_v).transpose(
                    0, 1).contiguous().view(n_batch, self.h, pos_v.size(0),
                                            self.d_k))
                x = x + attn_v
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h_k * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)


class MultiSequential(torch.nn.Sequential):
    """Multi-input multi-output torch.nn.Sequential"""

    @torch.jit.ignore
    def forward(self, *args):
        """Forward method implementation."""
        for m in self:
            args = m(*args)
        return args


def get_offset(input_layer: str, time_reduction: int):
    """Get an offset. We will use the offset for determining #frames of a 
    subsampled feature.

    Args:
        input_layer (str): Type of an input layer
        time_reduction (int): time reduction factor for downsampling a feature
    Returns:
        int: offset
    """
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 4:
        return 3
    if input_layer in ("conv2d", ) and time_reduction == 6:
        return 1
    if input_layer in ("conv2d", "nemo_conv") and time_reduction == 8:
        return 7
    return 0


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is 
    longer than max_seq_len, this function unfold it to a 
    (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )
    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad
