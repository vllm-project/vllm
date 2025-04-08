"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from vllm.transformers_utils.qwen2_code2wav_dit.model.spk_encoder import ECAPA_TDNN
from vllm.transformers_utils.qwen2_code2wav_dit.model.dit_modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

# Text embedding
class CodecEmbedding(nn.Module):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, codec: int["b nc"], seq_len, drop_text=False):
        if drop_text:
            codec = torch.zeros_like(codec)
        codec = self.codec_embed(codec)
        codec = torch.repeat_interleave(codec, repeats=self.repeats, dim=1)
        return codec

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim+ 128 + 192 + text_dim, out_dim) # 192 for x-vector
        self.spk_encoder = ECAPA_TDNN(80, 128,
                                        channels=[256, 256, 256, 256, 768],
                                        kernel_sizes=[5, 3, 3, 3, 1],
                                        dilations=[1, 2, 3, 4, 1],
                                        attention_channels=64,
                                        res2net_scale=2,
                                        se_channels=64,
                                        global_context=True,
                                        batch_norm=False)
        # remove convposembedding for causal or block causal
        # self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"],spk: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)
            spk = torch.zeros_like(spk)
        cond = self.spk_encoder(cond).unsqueeze(1).repeat(1, x.size(1), 1)
        # import pdb; pdb.set_trace() 
        # print(x.shape,cond.shape,text_embed.shape)
        x = self.proj(torch.cat((x, cond, text_embed, spk), dim=-1))
        # x = self.conv_pos_embed(x) + x
        return x
    
    def fast_forward(self, x, spk, cond, text_embed,text_embed_uncond):
        x = torch.cat([x,x],dim=0)
        spk = torch.cat([spk,torch.zeros_like(spk)],dim=0)
        cond = torch.cat([cond,torch.rand_like(cond)],dim=0)
        cond = self.spk_encoder(cond).unsqueeze(1).repeat(1, x.size(1), 1)
        text_emb = torch.cat([text_embed,text_embed_uncond],dim=0)
        x = self.proj(torch.cat((x, cond, text_emb, spk), dim=-1))

        return x



# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        use_codec=False, 
        attn_processor="",
        repeats=2
    ):
        super().__init__()
        self.repeats = repeats
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        if not use_codec:
            self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        else:
            self.text_embed = CodecEmbedding(text_num_embeds, text_dim, repeats=repeats)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth
        if attn_processor == "stream_block_sr":
            attn_processor_0 = 'stream_block_sr_00'
            attn_processor_1 = 'stream_block_sr_10'
            attn_processor_2 = 'stream_block_sr_01'
            self.transformer_blocks = nn.ModuleList()
            for i in range(depth):
                if i == 0 or i == 20:
                    attn_processor_in = attn_processor_1
                elif i == 10: 
                    attn_processor_in = attn_processor_2
                else:
                    attn_processor_in = attn_processor_0
                self.transformer_blocks.append(
                    DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, attn_processor=attn_processor_in)
                )
        elif attn_processor == "stream_block_sr_low":
            attn_processor_0 = 'stream_block_sr_00'
            attn_processor_1 = 'stream_block_sr_11'
            attn_processor_2 = 'stream_block_sr_10'
            self.transformer_blocks = nn.ModuleList()
            for i in range(depth):
                if i == 0:
                    attn_processor_in = attn_processor_1
                elif i == 10 or i == 20: 
                    attn_processor_in = attn_processor_2
                else:
                    attn_processor_in = attn_processor_0
                self.transformer_blocks.append(
                    DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, attn_processor=attn_processor_in)
                )
        elif attn_processor == 'stream_block':
            attn_processor_0 = 'stream_block_0'
            attn_processor_1 = 'stream_block_1'
            self.transformer_blocks = nn.ModuleList(
                [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, attn_processor=attn_processor_1 if i%5==0 else attn_processor_0) for i in range(depth)]
            )
        elif attn_processor == "stream_block_8_L_4":
            attn_processor_0 = 'stream_block_8_00'
            attn_processor_1 = 'stream_block_8_10'
            attn_processor_2 = 'stream_block_8_01'
            self.transformer_blocks = nn.ModuleList()
            for i in range(depth):
                if i == 0 or i == 8 or i == 24 or i == 30:
                    attn_processor_in = attn_processor_1
                elif i == 15:
                    attn_processor_in = attn_processor_2
                else:
                    attn_processor_in = attn_processor_0
                self.transformer_blocks.append(
                    DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, attn_processor=attn_processor_in)
                )
        else:
            self.transformer_blocks = nn.ModuleList(
                [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout, attn_processor=attn_processor) for i in range(depth)]
            )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        spk: float["b n d"],  # spk embedding  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        # print(spk.dtype, cond.dtype, text_embed.dtype)
        # import pdb; pdb.set_trace()
        x = self.input_embed(x, spk, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output

    @torch.no_grad()
    def fast_forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        spk: float["b n d"],  # spk embedding  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
    ) -> float["b n d"]:
        batch, seq_len = x.shape[0]*2, x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        text_embed_uncond = self.text_embed(text, seq_len, drop_text=True)
        # print(spk.dtype, cond.dtype, text_embed.dtype)
        # import pdb; pdb.set_trace()
        x = self.input_embed.fast_forward(x, spk, cond, text_embed, text_embed_uncond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
    