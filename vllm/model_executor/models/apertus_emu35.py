# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from Emu3.5 (Copyright Swiss AI Initiative).
# Licensed under the Apache License, Version 2.0 (the "License").

import os.path as osp
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import einsum


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw
        h_ = torch.bmm(v, w_)  # b, c,hw
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    ),
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class IndexPropagationQuantize(nn.Module):

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        remap: str | None = None,
        unknown_index: str | int = "random",
        l2_normalize: bool = False,
    ):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
        else:
            self.re_embed = n_e

        self.l2_normalize = l2_normalize

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device,
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, float, tuple[None, None, torch.Tensor]]:
        # z: [b, d, h, w]
        # embed.weight: [n, d]
        if self.l2_normalize:
            z = F.normalize(z, dim=1)
            embedding = F.normalize(self.embedding.weight, dim=1)
        else:
            embedding = self.embedding.weight

        logits = einsum("b d h w, n d -> b n h w", z, embedding)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        dim = 1
        ind = logits.argmax(dim=dim, keepdim=True)
        hard_one_hot = torch.zeros_like(
            logits,
            memory_format=torch.legacy_contiguous_format,
        ).scatter_(dim, ind, 1.0)

        z_q = einsum("b n h w, n d -> b d h w", hard_one_hot, embedding)

        ind = torch.flatten(ind)
        if self.remap is not None:
            ind = ind.reshape(z.shape[0], -1)
            ind = self.remap_to_used(ind)
            ind = ind.reshape(-1, 1)

        return z_q, 0.0, (None, None, ind)

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            assert shape is not None
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class IBQ(nn.Module):

    def __init__(
        self,
        ddconfig: dict[str, Any],
        n_embed: int,
        embed_dim: int,
        l2_normalize: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)

        self.quantize = IndexPropagationQuantize(
            n_e=n_embed,
            e_dim=embed_dim,
            l2_normalize=l2_normalize,
        )

        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, float, tuple[None, None, torch.Tensor]]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info


def build_vision_tokenizer(
    type: str,
    model_path: str,
    device: str = "cuda:0",
    vision_config: dict | None = None,
) -> IBQ:
    if type != "ibq":
        raise NotImplementedError(f"Unsupported vision tokenizer type: {type}")

    assert vision_config is not None, (
        "vision_config must be provided to build_vision_tokenizer"
    )

    # Check for safetensors or ckpt file
    safetensors_path = osp.join(model_path, "emu35_vison_tokenizer.safetensors")
    ckpt_path = osp.join(model_path, "model.ckpt")

    ddconfig = {
        "double_z": vision_config.get("double_z", False),
        "z_channels": vision_config.get("z_channels", 256),
        "resolution": vision_config.get("resolution", 256),
        "in_channels": vision_config.get("in_channels", 3),
        "out_ch": vision_config.get("out_ch", 3),
        "ch": vision_config.get("ch", 256),
        "ch_mult": tuple(vision_config.get("ch_mult", [1, 1, 2, 2, 4])),
        "num_res_blocks": vision_config.get("num_res_blocks", 4),
        "attn_resolutions": vision_config.get("attn_resolutions", [16]),
        "dropout": vision_config.get("dropout", 0.0),
    }

    cfg = {
        "ddconfig": ddconfig,
        "n_embed": vision_config.get("codebook_size", 131072),
        "embed_dim": vision_config.get("embed_dim", 256),
        "l2_normalize": vision_config.get("l2_normalize", False),
    }

    tokenizer = IBQ(**cfg)

    if osp.exists(safetensors_path):
        ckpt = load_file(safetensors_path, device="cpu")
        tokenizer.load_state_dict(ckpt, strict=False)
    elif osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        tokenizer.load_state_dict(ckpt, strict=False)
    else:
        raise FileNotFoundError(
            f"Apertus vision tokenizer checkpoint not found under {model_path}. "
            f"Expected either 'emu35_vison_tokenizer.safetensors' or 'model.ckpt'.",
        )

    tokenizer.eval().to(device)
    return tokenizer
