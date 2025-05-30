# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Code copied from Microsoft/MoE by Jacob Platin (jacobplatin@microsoft.com)
# but implemented by the Phi-Speech team
#!/usr/bin/env python3
import abc
import math
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel)
from transformers import PretrainedConfig

from vllm.model_executor.models.phi4mm_utils import (
    AbsolutePositionalEncoding, ConvModule, FeedForward, MeanVarianceNormLayer,
    MultiHeadedAttention, MultiSequential, NemoConvSubsampling,
    T5RelativeAttentionLogitBias, adaptive_enc_mask, get_offset, unfold_tensor)

_AUDIO_PLACEHOLDER_TOKEN_ID = 200011  # <|endoftext11|>


class ConformerEncoderLayer(nn.Module):
    """ConformerEncoder Layer module.
    for more details see conformer paper:
        https://arxiv.org/abs/2005.08100
    This module implement the Conformer block layer.

    Args:
        d_model: int
            attention dim.
        ext_pw_out_channel: int
            if > 0, ext_pw_out_channel is a dim channel size
             for the last pointwise conv after swish activation.
        depthwise_seperable_out_channel: int
            if set different to 0, the number of 
             depthwise_seperable_out_channel will be used as a 
             channel_out of the second conv1d layer. 
             otherwise, it equal to 0, the second conv1d layer is skipped.
        depthwise_multiplier: int
            number of input_dim channels duplication. this value
             will be used to compute the hidden channels of the Conv1D.
        n_head: int
            the number of heads for multihead attention module.
        d_ffn: int
            output size of the feed_forward blocks.
        ext_pw_kernel_size: int
            kernel size of the conv pointwise of the conformer.
        kernel_size: int
            kernel size.
        dropout_rate: float
            dropout rate.
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation
            in ConvModule layer of the conformer.
            default False
        activation: str, optional
            activation function name,
            one of ["relu", "swish", "sigmoid"],
            sigmoid activation is only used with "glu_in_fnn=True",
            default "relu".
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
            default 0.
        chunk_size: int, optional
            chunk_size for cnn. default 18
        conv_activation: str, optional
            activation function used in ConvModule part
            of the conformer, default "relu".
        conv_glu_type: str, optional
            activation function used for the glu inside
            the ConvModule part of the conformer.
            default: "sigmoid".
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU.
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        attention_inner_dim: int, optional
            if equal to -1, attention dim for linears k/q/v is
            equal to d_model. otherwise attention_inner_dim is used.
            default -1.
        attention_glu_type: str, optional
            activation function for glu used in the multihead attention,
             default "swish".
        activation_checkpointing: str, optional
            a dictionarry of {"module","interval","offload"}, where
                "module": str
                    accept ["transformer", "attention"] to select
                    which module should do activation checkpointing.
                "interval": int, default 1,
                    interval of applying activation checkpointing,
                    interval = 1 means that we apply checkpointing
                    on every layer (if activation), otherwise,
                    we apply it every x interval.
                "offload": bool, default False,
                    if set to True, we offload activation to cpu and
                    reload it during backward, otherwise,
                    we recalculate activation in backward.
            default "".
        export: bool, optional
            if set to True, it remove the padding from convolutional layers
             and allow the onnx conversion for inference.
              default False.
        use_pt_scaled_dot_product_attention: bool, optional
            if set to True, use pytorch's scaled dot product attention 
            implementation in training.
        attn_group_sizes: int, optional
            the number of groups to use for attention, default 1 
            (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attn_group_sizes < attention_heads = Grouped-Query Attention
            attn_group_sizes = attenion_heads = Multi-Query Attention
    """

    def __init__(
        self,
        d_model=512,
        ext_pw_out_channel=0,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        n_head=4,
        d_ffn=2048,
        ext_pw_kernel_size=1,
        kernel_size=3,
        dropout_rate=0.1,
        causal=False,
        batch_norm=False,
        activation="relu",
        chunk_se=0,
        chunk_size=18,
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_inner_dim=-1,
        attention_glu_type="swish",
        activation_checkpointing="",
        export=False,
        use_pt_scaled_dot_product_attention=False,
        attn_group_sizes: int = 1,
    ):
        super().__init__()

        self.feed_forward_in = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.self_attn = MultiHeadedAttention(
            n_head,
            d_model,
            dropout_rate,
            attention_inner_dim,
            attention_glu_type,
            bias_in_glu,
            use_pt_scaled_dot_product_attention=
            use_pt_scaled_dot_product_attention,
            group_size=attn_group_sizes,
        )
        self.conv = ConvModule(
            d_model,
            ext_pw_out_channel,
            depthwise_seperable_out_channel,
            ext_pw_kernel_size,
            kernel_size,
            depthwise_multiplier,
            dropout_rate,
            causal,
            batch_norm,
            chunk_se,
            chunk_size,
            conv_activation,
            conv_glu_type,
            bias_in_glu,
            linear_glu_in_convm,
            export=export,
        )

        self.feed_forward_out = FeedForward(
            d_model=d_model,
            d_inner=d_ffn,
            dropout_rate=dropout_rate,
            activation=activation,
            bias_in_glu=bias_in_glu,
        )

        self.layer_norm_att = nn.LayerNorm(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        pos_k,
        pos_v,
        mask,
        relative_attention_bias: Optional[Tensor] = None,
    ):
        """ConformerEncoder forward.

        Args:
            x: torch.Tensor
                input feature of shape (batch, max_time_in, size)
            pos_k: torch.Tensor
                positional key embedding.
            mask: torch.Tensor
                mask for x (batch, max_time_in)
            relative_attention_bias: Optional[torch.Tensor]
                bias added to attention logits w.r.t. relative positions 
                (1, n_head, time1, time2)
        """
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(
            norm_x,
            norm_x,
            norm_x,
            pos_k,
            pos_v,
            mask,
            relative_attention_bias=relative_attention_bias,
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out, pos_k, pos_v, mask


class TransformerEncoderBase(abc.ABC, nn.Module):
    """The Base class for Transformer based encoders

    Please set causal = True in streaming model
    Args:
        input_size: int
            input feature dimension.
        chunk_size: int, list(int)
            Number of frames for each chunk
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training
            Some examples for the 2 cases:
            chunk_size = 12
            chunk_size = [6, 8, 12, 24]
        left_chunk: int, list(int)
            Number of chunks used for masking in streaming mode.
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training. When
            chunk_size is a list, left_chunk must be a list with same length.
            Some examples for the 2 cases:
            left_chunk = 6
            left_chunk = [12, 9, 6, 3]
        attention_dim: int, optional
            attention dimension. default 256.
        attention_heads: int, optional
            the number of heads. default 4
        input_layer: str, optional
            input layer type before Conformer,
            one of ["linear", "conv2d", "custom", "vgg2l", "embed"],
            default "conv2d"
        cnn_out: int, optional
            the number of CNN channels before Conformer.
            default -1.
        cnn_layer_norm: bool, optional
            layer norm between Conformer and the first CNN.
            default False.
        time_reduction: int, optional
            time reduction factor
            default 4
        dropout_rate: float, optional
            dropout rate. default 0.1
        padding_idx: int, optional
            padding index for input_layer=embed
            default -1
        relative_attention_bias_args: dict, optional
            use more efficient scalar bias-based relative multihead attention
            (Q*K^T + B) implemented in cmb.basics.embedding.
            [T5/ALiBi]RelativeAttentionLogitBias
            usage: relative_attention_bias_args={"type": t5/alibi}
            additional method-specific arguments can be provided (see 
            transformer_base.py)
        positional_dropout_rate: float, optional
            dropout rate after positional encoding. default 0.0
        nemo_conv_settings: dict, optional
            A dictionary of settings for NeMo Subsampling.
            default None
        conv2d_extra_padding: str, optional
            Add extra padding in conv2d subsampling layers. Choices are
            (feat, feat_time, none, True).
            if True or feat_time, the extra padding is added into non full
            supraframe utts in batch.
            Default: none
        attention_group_size: int, optional
            the number of groups to use for attention, default 1 
            (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attention_group_size < attention_heads = Grouped-Query 
            Attention
            attention_group_size = attenion_heads = Multi-Query Attention
    """

    def __init__(
        self,
        input_size,
        chunk_size,
        left_chunk,
        attention_dim=256,
        attention_heads=4,
        input_layer="nemo_conv",
        cnn_out=-1,
        cnn_layer_norm=False,
        time_reduction=4,
        dropout_rate=0.0,
        padding_idx=-1,
        relative_attention_bias_args=None,
        positional_dropout_rate=0.0,
        nemo_conv_settings=None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none",
                                      True] = "none",
        attention_group_size=1,
        encoder_embedding_config=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_layer = input_layer
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.attention_dim = attention_dim
        self.num_heads = attention_heads
        self.attention_group_size = attention_group_size
        self.time_reduction = time_reduction
        self.nemo_conv_settings = nemo_conv_settings
        self.encoder_embedding_config = encoder_embedding_config

        if self.input_layer == "nemo_conv":
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.time_reduction,
                "feat_in": input_size,
                "feat_out": attention_dim,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            # Override any of the defaults with the incoming, user settings
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (
                        i not in nemo_conv_settings
                    ), "{i} should be specified outside of the NeMo dictionary"

            self.embed = NemoConvSubsampling(**default_nemo_conv_settings, )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.pos_emb = AbsolutePositionalEncoding(attention_dim,
                                                  positional_dropout_rate)

        self.relative_attention_bias_type = (
            relative_attention_bias_args.get("type")
            if relative_attention_bias_args else None)
        if self.relative_attention_bias_type == "t5":
            assert (self.num_heads % self.attention_group_size == 0
                    ), "attention_group_size must divide n_head"
            self.relative_attention_bias_layer = T5RelativeAttentionLogitBias(
                self.num_heads // self.attention_group_size,
                max_distance=relative_attention_bias_args.get(
                    "t5_bias_max_distance", 1000),
                symmetric=relative_attention_bias_args.get(
                    "t5_bias_symmetric", False),
            )
        else:
            raise NotImplementedError

        self.encoder_embedding = MeanVarianceNormLayer(
            self.encoder_embedding_config["input_size"])

    def compute_lens_change(self, feature_lens):
        """feature_lens: int
        return updated feature lens.

        This used to return a different lambda function for each case that 
        computed the right thing.  That does not work within Torchscript. 
        If you really need this to be faster, create nn.Module()-s for all
        the cases and return one of them.  Torchscript does support that.
        """
        if self.input_layer == "nemo_conv":
            # Handle the special causal case
            subsampling_causal_cond = self.nemo_conv_settings.get(
                "subsampling", "dw_striding") in [
                    "dw_striding",
                    "striding",
                    "striding_conv1d",
                ]
            is_causal = self.nemo_conv_settings.get("is_causal", False)
            if is_causal and subsampling_causal_cond:
                lens_change = (torch.ceil(feature_lens /
                                          self.time_reduction).long()
                               if isinstance(feature_lens, Tensor) else
                               math.ceil(feature_lens / self.time_reduction))
                feature_lens_remainder = feature_lens % self.time_reduction
                if isinstance(feature_lens, Tensor):
                    lens_change[feature_lens_remainder != 1] += 1
                elif feature_lens_remainder != 1:
                    lens_change += 1
                return lens_change
            ceil_func = (math.ceil
                         if isinstance(feature_lens, int) else torch.ceil)
            return ceil_func(feature_lens / self.time_reduction)

    @abc.abstractmethod
    def forward(self):
        """Abstract forward method implementation."""

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""

        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(
                torch.randint(low=0, high=len(chunk_size), size=(1, )))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError(
                    "Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError(
                    "The length of left_chunk must be the same as length of "\
                        "chunk_size."
                )
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def _get_embed_class(self, embed):
        # pylint: disable=protected-access
        is_embed_using_act_chkpt = isinstance(embed, CheckpointWrapper)
        is_embed_fsdp_wrapped = isinstance(embed, FullyShardedDataParallel)
        embed_class = embed
        if is_embed_using_act_chkpt:
            embed_class = embed._checkpoint_wrapped_module
        if is_embed_fsdp_wrapped:
            embed_class = embed.module
        return embed_class

    def _forward_embeddings_core(self, input_tensor, masks):
        embed_class = self._get_embed_class(self.embed)
        assert isinstance(embed_class, NemoConvSubsampling)
        input_tensor, masks = self.embed(input_tensor, masks)
        return input_tensor, masks

    def _position_embedding(self, input_tensor):
        pos_k = None
        pos_v = None
        if self.relative_attention_bias_layer is None:
            input_tensor = self.pos_emb(
                input_tensor)  # default to add abs sinusoid embedding
        return pos_k, pos_v

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = \
            self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)

        enc_streaming_mask = (adaptive_enc_mask(
            seq_len, chunk_start_idx,
            left_window=left_chunk_train_eff).unsqueeze(0).expand(
                [batch_size, -1, -1]))
        return enc_streaming_mask

    def forward_embeddings(self,
                           xs_pad,
                           masks,
                           chunk_size_nc=None,
                           left_chunk_nc=None):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
            chunk_size_nc: (optional, default is None) chunk size for 
                            non-causal layers
            left_chunk_nc: (optional, default is None) # of left chunks for
                            non-causal layers
        """
        # pylint: disable=R0915
        # get new lens.
        seq_len = int(self.compute_lens_change(xs_pad.shape[1]))
        if seq_len <= 0:
            raise ValueError(
                f"""The sequence length after time reduction is invalid: 
                {seq_len}. Your input feature is too short. Consider 
                filtering out the very short sentence from data 
                loader""", )

        batch_size = xs_pad.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size,
                                                  self.chunk_size,
                                                  self.left_chunk)

        if xs_pad.is_cuda:
            enc_streaming_mask = enc_streaming_mask.cuda()
            xs_pad = xs_pad.cuda()

        input_tensor = xs_pad
        input_tensor, masks = self._forward_embeddings_core(
            input_tensor, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        if chunk_size_nc is not None:
            enc_streaming_mask_nc = self._streaming_mask(
                seq_len, batch_size, chunk_size_nc, left_chunk_nc)
            if xs_pad.is_cuda:
                enc_streaming_mask_nc = enc_streaming_mask_nc.cuda()
            if masks is not None:
                hs_mask_nc = masks & enc_streaming_mask_nc
            else:
                hs_mask_nc = enc_streaming_mask_nc
        else:
            hs_mask_nc = None

        pos_k, pos_v = self._position_embedding(input_tensor)

        if chunk_size_nc is None:
            return input_tensor, pos_k, pos_v, hs_mask, masks
        return input_tensor, pos_k, pos_v, hs_mask, masks, hs_mask_nc

    def get_offset(self):
        """Returns offset used when retaining inputs for decoding.

        This is essentially, how many additional frames have to be added to
        the front-end CNN input to ensure it can produce a single output.
        So if the "padding" parameter is 0, typically offset will be > 0.
        """
        return get_offset(self.input_layer, self.time_reduction)


class ConformerEncoder(TransformerEncoderBase):
    """ConformerEncoder module.
    see original paper for more details:
        https://arxiv.org/abs/2005.08100

    Please set causal = True in streaming model
    Args:
        input_size: int
            input feature dimension.
        chunk_size: int, list(int)
            Number of frames for each chunk
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training
            Some examples for the 2 cases:
            chunk_size = 12
            chunk_size = [6, 8, 12, 24]
        left_chunk: int, list(int)
            Number of chunks used for masking in streaming mode.
            This variable can take 2 forms:
            int:  Used for inference, or single chunk size training
            list(int) : Used only for variable chunk size training. When
            chunk_size is a list, left_chunk must be a list with same length.
            Some examples for the 2 cases:
            left_chunk = 6
            left_chunk = [12, 9, 6, 3]
        left_chunk: int
            number of chunks used for masking in streaming mode.
        num_lang: int
            This parameter is used to store the number of languages in the 
            lang_dict, only used for multiseed/multilingual models. 
            default None.
        attention_dim: int, optional
            attention dimension. default 256.
        attention_heads: int, optional
            the number of heads. default 4
        linear_units:
            the number of units of position-wise feed forward.
            default 2048
        num_block:
            number of Transformer layer. default 6
        dropout_rate: float, optional
            dropout rate. default 0.1
        input_layer: str, optional
            input layer type before Conformer,
            one of ["linear", "conv2d", "custom", "vgg2l", "embed"],
            default "conv2d"
        causal: bool, optional
            if set to True, convolution have no access
             to future frames. default False.
        batch_norm: bool, optional
            if set to True, apply batchnorm before activation
            in ConvModule layer of the conformer.
            default False
        cnn_out: int, optional
            the number of CNN channels before Conformer.
            default -1.
        cnn_layer_norm: bool, optional
            layer norm between Conformer and the first CNN.
            default False.
        ext_pw_out_channel: int, optional
            the number of channel for CNN
            before depthwise_seperable_CNN.
            If 0 then use linear. default 0.
        ext_pw_kernel_size: int, optional
            kernel size of N before depthwise_seperable_CNN.
            only work for ext_pw_out_channel > 0.
            default 1
        depthwise_seperable_out_channel: int, optional
            the number of channel for
            depthwise_seperable_CNN.
            default 256.
        depthwise_multiplier: int, optional
            the number of multiplier for
            depthwise_seperable_CNN.
            default 1.
        chunk_se: int, optional
            0 for offline SE.
            1 for streaming SE, where mean is computed
             by accumulated history until current chunk_se.
            2 for streaming SE, where mean is computed
             by only the current chunk.
            default 0.
        kernel_size: int, optional
            the number of kernels for depthwise_seperable_CNN.
            default 3.
        activation: str, optional
            FeedForward block activation.
            one of ["relu", "swish", "sigmoid"]
            default "relu".
        conv_activation: str, optional
            activation function used in ConvModule part
            of the conformer, default "relu".
        conv_glu_type: str, optional
            activation used use glu in depthwise_seperable_CNN,
            default "sigmoid"
        bias_in_glu: bool, optional
            if set to True, use additive bias in the weight module
             before GLU. default True
        linear_glu_in_convm: bool, optional
            if set to True, use GLULinear module,
             otherwise, used GLUPointWiseConv module.
              default to False.
        attention_glu_type: str
            only work for glu_in_attention !=0
            default "swish".
        export: bool, optional
            if set to True, it remove the padding from convolutional layers
             and allow the onnx conversion for inference.
              default False.
        activation_checkpointing: str, optional
            a dictionarry of {"module","interval","offload"}, where
                "module": str
                    accept ["transformer", "attention"] to select
                    which module should do activation checkpointing.
                "interval": int, default 1,
                    interval of applying activation checkpointing,
                    interval = 1 means that we apply checkpointing
                    on every layer (if activation), otherwise,
                    we apply it every x interval.
                "offload": bool, default False,
                    if set to True, we offload activation to cpu and
                    reload it during backward, otherwise,
                    we recalculate activation in backward.
            default "".
        extra_layer_output_idx: int
            the layer index to be exposed.
        relative_attention_bias_args: dict, optional
            use more efficient scalar bias-based relative multihead attention 
            (Q*K^T + B) implemented in cmb.basics.embedding.
            [T5/ALiBi]RelativeAttentionLogitBias
            usage: relative_attention_bias_args={"type": t5/alibi}
            additional method-specific arguments can be provided (see 
            transformer_base.py)
        time_reduction: int optional
            time reduction factor
            default 4
        use_pt_scaled_dot_product_attention: whether to use pytorch scaled 
            dot product attention in training.
            Default: False
        nemo_conv_settings: dict, optional
            A dictionary of settings for NeMo Subsampling.
            default: None
            usage: nemo_conv_settings=
                {
                    "subsampling":
                    dw_striding/striding/dw_striding_conv1d/striding_conv1d,
                    "conv_channels": int,
                    "subsampling_conv_chunking_factor": int,
                    "is_causal": True/False
                }
        conv2d_extra_padding: str, optional
            Add extra padding in conv2d subsampling layers. Choices are
            (feat, feat_time, none, True)
            Default: none
        replication_pad_for_subsample_embedding:  For batched-streaming 
            decoding, use "replication" padding for the cache at start of
            utterance.
            Default: False
        attention_group_size: int, optional
            the number of groups to use for attention, default 1 
            (Multi-Head Attention),
            1 = typical Multi-Head Attention,
            1 < attention_group_size < attention_heads = Grouped-Query
            Attention
            attention_group_size = attenion_heads = Multi-Query Attention
    """

    extra_multi_layer_output_idxs: list[int]

    def __init__(  # pylint: disable-all
        self,
        input_size,
        chunk_size,
        left_chunk,
        num_lang=None,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        input_layer="nemo_conv",
        causal=True,
        batch_norm=False,
        cnn_out=-1,
        cnn_layer_norm=False,
        ext_pw_out_channel=0,
        ext_pw_kernel_size=1,
        depthwise_seperable_out_channel=256,
        depthwise_multiplier=1,
        chunk_se=0,
        kernel_size=3,
        activation="relu",
        conv_activation="relu",
        conv_glu_type="sigmoid",
        bias_in_glu=True,
        linear_glu_in_convm=False,
        attention_glu_type="swish",
        export=False,
        extra_layer_output_idx=-1,
        extra_multi_layer_output_idxs=[],  # noqa
        activation_checkpointing="",
        relative_attention_bias_args=None,
        time_reduction=4,
        use_pt_scaled_dot_product_attention=False,
        nemo_conv_settings=None,
        conv2d_extra_padding: Literal["feat", "feat_time", "none",
                                      True] = "none",
        replication_pad_for_subsample_embedding=False,
        attention_group_size=1,
        encoder_embedding_config=None,
    ):
        super().__init__(
            input_size,
            chunk_size,
            left_chunk,
            attention_dim,
            attention_heads,
            input_layer,
            cnn_out,
            cnn_layer_norm,
            time_reduction,
            dropout_rate=dropout_rate,
            relative_attention_bias_args=relative_attention_bias_args,
            positional_dropout_rate=0.0,
            nemo_conv_settings=nemo_conv_settings,
            conv2d_extra_padding=conv2d_extra_padding,
            attention_group_size=attention_group_size,
            encoder_embedding_config=encoder_embedding_config,
        )
        self.num_blocks = num_blocks
        self.num_lang = num_lang
        self.kernel_size = kernel_size
        self.replication_pad_for_subsample_embedding: bool = (
            replication_pad_for_subsample_embedding)
        assert (self.num_heads % attention_group_size == 0
                ), "attention_group_size must divide n_head"
        self.num_heads_k = self.num_heads // attention_group_size

        self.encoders = MultiSequential(*[
            ConformerEncoderLayer(
                d_model=attention_dim,
                ext_pw_out_channel=ext_pw_out_channel,
                depthwise_seperable_out_channel=depthwise_seperable_out_channel,
                depthwise_multiplier=depthwise_multiplier,
                n_head=attention_heads,
                d_ffn=linear_units,
                ext_pw_kernel_size=ext_pw_kernel_size,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                causal=causal,
                batch_norm=batch_norm,
                activation=activation,
                chunk_se=chunk_se,
                chunk_size=chunk_size,
                conv_activation=conv_activation,
                conv_glu_type=conv_glu_type,
                bias_in_glu=bias_in_glu,
                linear_glu_in_convm=linear_glu_in_convm,
                attention_glu_type=attention_glu_type,
                activation_checkpointing=activation_checkpointing,
                export=export,
                use_pt_scaled_dot_product_attention=
                use_pt_scaled_dot_product_attention,
                attn_group_sizes=attention_group_size,
            ) for _ in range(num_blocks)
        ])
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        # Make a zeros scalar we can use in get_initial_state to determine
        # the device and the needed dtype:
        self.register_buffer("dev_type", torch.zeros(()), persistent=False)

    def init_relative_attention_bias(self, input_tensor):
        if self.relative_attention_bias_layer:
            return self.relative_attention_bias_layer(input_tensor)

    def calculate_hs_mask(self, xs_pad, device, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size,
                                                  self.chunk_size,
                                                  self.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = (torch.arange(0, max_audio_length,
                                 device=device).expand(padding_length.size(0),
                                                       -1)
                    < padding_length.unsqueeze(1))
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    @torch.jit.ignore
    def forward(self, xs_pad, masks):
        """Conformer Forward function

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                post-embedding input lengths
        """
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, pos_k, pos_v, hs_mask, masks = self.forward_embeddings(
            xs_pad, masks)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500  #maximum position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks
            # of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple
            # of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = F.pad(input_tensor,
                                         (0, 0, 0, chunk_pad_size), "constant",
                                         0)
                input_tensor = input_tensor_pad.to(input_tensor.device)
            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask
                # did not consider extra pad
                subsampled_pad_mask = masks.squeeze(
                    1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant",
                    False)  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = \
                    extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(
                    -1).bool()  # unfold op does not support bool tensor
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(
                input_tensor, input_tensor.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        # layer_emb = None

        relative_attention_bias = self.init_relative_attention_bias(
            input_tensor)

        _simplified_path = (self.extra_layer_output_idx == -1
                            and relative_attention_bias is None)

        if _simplified_path:
            input_tensor, *_ = self.encoders(input_tensor, pos_k, pos_v,
                                             hs_mask)
        else:
            for i, layer in enumerate(self.encoders):
                input_tensor, _, _, _ = layer(
                    input_tensor,
                    pos_k,
                    pos_v,
                    hs_mask,
                    relative_attention_bias=relative_attention_bias,
                )

                # if i == self.extra_layer_output_idx:
                #     layer_emb = input_tensor

        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]

        return input_tensor, masks  # , layer_emb


class WindowQformer(nn.Module):
    """Window-level Qformer"""

    def __init__(
        self,
        window_size: int = 8,
        num_queries: int = 1,
        num_blocks: int = 2,
        attention_dim: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        dropout_rate: float = 0.0,
        normalize_before: bool = True,
    ):
        super().__init__()

        self.decoders = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=attention_dim,
                nhead=attention_heads,
                dim_feedforward=linear_units,
                dropout=dropout_rate,
                activation="relu",
                batch_first=True,
                norm_first=normalize_before,  # TODO need to verify
            ) for _ in range(num_blocks)
        ])

        self.queries = nn.Parameter(torch.zeros(1, num_queries, attention_dim))
        self.after_norm = (nn.LayerNorm(attention_dim, eps=1e-12)
                           if normalize_before else None)
        self.window_size = window_size

    def forward(self, audio_embed, mask, embed_len=None):
        """forward decoder"""
        # audio_embed: N x T x D => N x D x T

        audio_embed = audio_embed.transpose(1, 2)
        # audio_embed: N x D x 1 x T => N x DK x T'
        padding = audio_embed.shape[-1] % self.window_size
        if padding > 0:
            audio_embed = F.pad(audio_embed, (0, self.window_size - padding),
                                "constant", 0)

        embed_chunk = F.unfold(
            audio_embed[..., None, :],
            kernel_size=(1, self.window_size),
            stride=(1, self.window_size),
        )
        bsz, _, slen = embed_chunk.shape
        # N x D x K x T'
        embed_chunk = embed_chunk.view(bsz, -1, self.window_size, slen)
        # N x T' x K x D
        embed_chunk = embed_chunk.transpose(1, 3).contiguous()
        # NT' x K x D
        embed_chunk = embed_chunk.view(bsz * slen, self.window_size, -1)
        # NT' x 1 x D
        q = self.queries.expand(bsz * slen, -1, -1)
        for layer in self.decoders:
            q = layer(tgt=q,
                      memory=embed_chunk,
                      tgt_mask=None,
                      memory_mask=mask)

        if self.after_norm is not None:
            q = self.after_norm(q)

        if embed_len is not None:
            embed_len = embed_len // self.window_size
        # N x T' x D
        out = q.view(bsz, slen, -1)

        return out, embed_len


class AudioEmbedding(nn.Module):
    """Image embedding."""

    def __init__(self, config: PretrainedConfig, **kwargs) -> None:
        super().__init__()
        self.config = config
        # n_embed or hidden_size for text LM
        hidden_size = (config.n_embd
                       if hasattr(config, "n_embd") else config.hidden_size)

        # self.wte = nn.Embedding(config.vocab_size, hidden_size)

        audio_dim_out = (
            None  # Set this variable according to the actual audio processor
        )
        self.layer_idx = -2

        if (isinstance(config.audio_processor, dict)
                and config.audio_processor.get("name", None) == "cascades"):
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None
            self.encoder = ConformerEncoder(**encoder_config)

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError("")

        assert (audio_dim_out
                is not None), "Remember to set values for audio_dim_out"
        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        self.freeze_audio_processor = kwargs.get("freeze_audio_processor",
                                                 False)

        self.downsample_rate = kwargs.get("downsample_rate", 1)

        if kwargs.get("use_qformer", False):
            qformer_config = kwargs.get("qformer_config", {})
            qformer_config["attention_dim"] = audio_dim_out
            self.qformer = WindowQformer(**qformer_config)
        else:
            self.qformer = None

        if kwargs.get("use_conv_downsample", False):
            assert (self.qformer is None
                    ), "don't support use qformer and conv downsample together"
            nemo_conv_settings = kwargs.get("nemo_conv_settings", {})
            default_nemo_conv_settings = {
                "subsampling": "dw_striding",
                "subsampling_factor": self.downsample_rate,
                "feat_in": audio_dim_out,
                "feat_out": audio_dim_out,
                "conv_channels": 256,
                "subsampling_conv_chunking_factor": 1,
                "activation": nn.ReLU(),
                "is_causal": False,
            }
            # Override any of the defaults with the incoming, user settings
            if nemo_conv_settings:
                default_nemo_conv_settings.update(nemo_conv_settings)
                for i in ["subsampling_factor", "feat_in", "feat_out"]:
                    assert (
                        i not in nemo_conv_settings
                    ), "{i} should be specified outside of the NeMo dictionary"

            self.conv_ds = NemoConvSubsampling(**default_nemo_conv_settings, )
        else:
            self.conv_ds = None

        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == "mlp":
            # follow llava-v1.5's implementation
            # (do not use image_projection and image_proj_norm)
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = (1 if (self.qformer or self.conv_ds)
                                           else self.downsample_rate)
            layers = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate,
                          dim_projection)
            ]
            for _ in range(1, depth):
                layers.extend(
                    [nn.GELU(),
                     nn.Linear(dim_projection, dim_projection)])
            self.audio_projection = nn.Sequential(*layers)
            # NOTE vision-speech tasks use a separate projection layer
            layers = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate,
                          dim_projection)
            ]
            for _ in range(1, depth):
                layers.extend(
                    [nn.GELU(),
                     nn.Linear(dim_projection, dim_projection)])
            self.audio_projection_for_vision = nn.Sequential(*layers)
        else:
            raise NotImplementedError(
                f"projection_cls = {projection_cls}, not implemented")

        # TODO: audio sequence compression - Qformer
        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def set_audio_embeds(self, input_embeds: torch.FloatTensor) -> None:
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self,
                              audio_embed_sizes: torch.LongTensor) -> None:
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self,
        input_embeds: torch.FloatTensor,
        audio_attention_mask: torch.Tensor = None,
        audio_projection_mode: str = "speech",
    ) -> torch.FloatTensor:
        """
        arguments:
            input_embeds: audio features (B, T, D)  B: num audios in a sequence
        """
        if self.freeze_audio_processor:
            with torch.no_grad():
                audio_features, masks = self.encoder(input_embeds,
                                                     audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds,
                                                 audio_attention_mask)

        if self.qformer is not None:
            audio_features, _ = self.qformer(audio_features, mask=None)

        if self.conv_ds is not None:
            if masks is not None:
                masks = masks.squeeze(1)

            audio_features, masks = self.conv_ds(audio_features, mask=masks)

        if self.linear_downsample_rate != 1:
            bs, seq_len, feat_dim = audio_features.size()
            padding = seq_len % self.linear_downsample_rate
            if padding > 0:
                audio_features = F.pad(
                    audio_features,
                    (0, 0, 0, self.linear_downsample_rate - padding),
                    "constant",
                    0,
                )

            seq_len = audio_features.size(1)
            audio_features = audio_features.view(
                bs,
                seq_len // self.linear_downsample_rate,
                feat_dim * self.linear_downsample_rate,
            )

        if audio_projection_mode == 'speech':
            audio_set_tensor = self.audio_projection(audio_features)
        elif audio_projection_mode == 'vision':
            audio_set_tensor = self.audio_projection_for_vision(audio_features)
        else:
            raise ValueError(
                f"audio_projection_mode = {audio_projection_mode} not "\
                    "implemented"
            )

        return audio_set_tensor

    def forward(
        self,
        audio_features: torch.FloatTensor,
        audio_attention_mask: torch.Tensor = None,
        audio_projection_mode: str = "speech",
    ) -> torch.FloatTensor:
        """
        arguments:
            audio_features: audio features (T, D)
        
        returns:
            audio_embeds: audio embeddings (num_audio_tokens, hidden_dim)
        """
        audio_embeds = self.get_audio_features(
            audio_features.unsqueeze(0),
            audio_attention_mask=audio_attention_mask,
            audio_projection_mode=audio_projection_mode,
        )
        return audio_embeds.squeeze(0)
