# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The HunYuan team.

import itertools
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.fusion.mm_input_norm import IdentityInputNorm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.transformers_utils.processors.hy_v4 import vision_token_count
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class HYV4VisionAttention(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        tp = 1 if use_dp else get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_dp,
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=use_dp,
        )
        self.attn = MMEncoderAttention(
            self.num_heads, self.head_dim, self.head_dim**-0.5, prefix=f"{prefix}.attn"
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True, enable_fp32_compute=True
        )

    def forward(
        self, x, cu_seqlens, max_seqlen, cos=None, sin=None, sequence_lengths=None
    ):
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        if cos is not None:
            shape = (*q.shape[:2], self.num_heads, self.head_dim)
            q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
            q, k = self.apply_rotary_emb(torch.cat((q, k), dim=0), cos, sin).chunk(
                2, dim=0
            )
        if self.attn.attn_backend == AttentionBackendEnum.FLASHINFER:
            q, k = q.contiguous(), k.contiguous()
        out = self.attn(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        return self.o_proj(out.reshape(*x.shape[:2], -1))[0]


class HYV4VisionMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_h_to_4h",
            disable_tp=use_dp,
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dense_4h_to_h",
            disable_tp=use_dp,
        )
        self.act_fn = get_act_fn(config.hidden_act)

    def forward(self, x):
        x = self.act_fn(self.dense_h_to_4h(x)[0])
        return self.dense_4h_to_h(x)[0]


class HYV4VisionBlock(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.self_attn = HYV4VisionAttention(
            config, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = HYV4VisionMLP(config, quant_config, f"{prefix}.mlp")

    def forward(self, x, **kwargs):
        x = x + self.self_attn(self.input_layernorm(x), **kwargs)
        return x + self.mlp(self.post_attention_layernorm(x))


class HYV4VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.position_edge = config.max_image_size // config.patch_size
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.position_embedding = nn.Embedding(
            self.position_edge**2 + 1, config.hidden_size, dtype=torch.float32
        )

    def _patch_conv(self, pixel_values: torch.Tensor) -> torch.Tensor:
        config = self.config
        patches = pixel_values.reshape(
            -1, config.num_channels, config.patch_size, config.patch_size
        )
        return self.patch_embedding(patches).flatten(2).squeeze(-1).unsqueeze(0)

    def compute_pos_embeds(
        self, grid_thw: list[list[int]], dtype: torch.dtype
    ) -> torch.Tensor:
        """Interpolated position embedding per patch, `(num_patches, hidden)`."""
        positions = (
            self.position_embedding.weight[1:]
            .float()
            .reshape(1, self.position_edge, self.position_edge, self.config.hidden_size)
            .permute(0, 3, 1, 2)
        )
        positional = []
        for t, h, w in grid_thw:
            resized = F.interpolate(
                positions,
                scale_factor=(
                    (h + 0.1) / self.position_edge,
                    (w + 0.1) / self.position_edge,
                ),
                mode=self.config.interpolate_mode,
                align_corners=False,
            )
            positional.append(
                resized.flatten(2).transpose(1, 2).to(dtype).repeat(1, t, 1)
            )
        return torch.cat(positional, dim=1).squeeze(0)

    def forward(self, pixel_values, grid_thw):
        patches = self._patch_conv(pixel_values)
        pos_embeds = self.compute_pos_embeds(grid_thw, patches.dtype)
        return patches + pos_embeds.unsqueeze(0)

    def forward_with_pos_embeds(
        self, pixel_values: torch.Tensor, pos_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Packed path: patch conv plus precomputed position embeddings."""
        return self._patch_conv(pixel_values) + pos_embeds.unsqueeze(0)


class HYV4VisionPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, output, merge = (
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
        )
        hidden = config.perceive_intermediate_size or dim * merge**2
        self.extra_mlp = config.projector_type is None
        if config.projector_type == "rmsnorm_mlp_adapter":
            self.norm = RMSNorm(dim, eps=config.rms_norm_eps)
            self.proj = nn.Sequential(
                nn.Conv2d(dim, hidden, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(hidden, output, 1),
            )
            newline_size = output
        elif config.projector_type is None:
            self.proj = nn.Sequential(
                nn.Conv2d(dim, dim * merge, merge, stride=merge),
                nn.GELU(),
                nn.Conv2d(dim * merge, hidden, 1),
            )
            self.mlp = nn.Linear(hidden, output)
            self.image_sep = nn.Parameter(torch.empty(output))
            newline_size = hidden
        else:
            raise ValueError(
                f"Unsupported HYV4 projector_type: {config.projector_type}"
            )
        if config.use_image_newline_token:
            self.image_newline = nn.Parameter(torch.empty(newline_size))
        if config.cat_extra_token:
            self.image_begin = nn.Parameter(torch.empty(output))
            self.image_end = nn.Parameter(torch.empty(output))
        self.before_rms = (
            RMSNorm(dim, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_pre_norm
            else None
        )
        self.after_rms = (
            RMSNorm(output, eps=config.rms_norm_eps)
            if self.extra_mlp and config.perceive_post_norm
            else None
        )

    def forward(self, x):
        config = self.config
        if not self.extra_mlp:
            x = self.norm(x)
        if self.before_rms is not None:
            x = self.before_rms(x)
        dtype = x.dtype
        x = self.proj(x.permute(0, 3, 1, 2).contiguous())
        if x.shape[0] > 1:
            x = F.avg_pool3d(
                x.permute(1, 0, 2, 3),
                (config.temporal_patch_size, 1, 1),
                stride=(config.temporal_patch_size, 1, 1),
            ).permute(1, 0, 2, 3)
            x = F.avg_pool2d(
                x, config.spatial_patch_size, stride=config.spatial_patch_size
            )
        b, c, h, _ = x.shape
        if config.use_image_newline_token:
            x = torch.cat(
                (
                    x,
                    self.image_newline.reshape(1, c, 1, 1).expand(b, c, h, 1).to(dtype),
                ),
                dim=-1,
            )
        x = x.flatten(2).transpose(1, 2)
        if self.extra_mlp:
            x = self.mlp(x)
        if config.cat_extra_token:
            x = torch.cat(
                (
                    self.image_begin.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                    x,
                    self.image_end.reshape(1, 1, -1).expand(b, 1, -1).to(dtype),
                ),
                dim=1,
            )
        x = x.reshape(-1, x.shape[-1])
        if self.after_rms is not None:
            x = self.after_rms(x)
        return x.contiguous()


class HYV4VisionTransformer(nn.Module):
    def __init__(
        self,
        config,
        quant_config=None,
        prefix="",
        input_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.input_norm = input_norm or IdentityInputNorm()
        self.embeddings = HYV4VisionPatchEmbed(config)
        self.layers = nn.ModuleList(
            [
                HYV4VisionBlock(config, quant_config, f"{prefix}.layers.{i}")
                for i in range(config.num_hidden_layers)
            ]
        )
        self.perceive = HYV4VisionPatchMerger(config)
        self.attn_backend = get_vit_attn_backend(
            head_size=config.hidden_size // config.num_attention_heads,
            dtype=torch.get_default_dtype(),
        )
        use_dp = is_vit_use_data_parallel(config.num_attention_heads)
        self.tp_size = 1 if use_dp else get_tensor_model_parallel_world_size()
        self.rotary_pos_emb = None
        if config.use_2d_rope:
            if config.rope_2d_percent != 1.0:
                raise ValueError("HYV4 vision supports only full 2D RoPE")
            self.rotary_pos_emb = get_rope(
                head_size=config.hidden_size // config.num_attention_heads,
                max_position=self.embeddings.position_edge + 1,
                is_neox_style=True,
                rope_parameters={
                    "rope_theta": config.rope_2d_theta,
                    "partial_rotary_factor": 0.5,
                },
                dtype=torch.float32,
            )

    @property
    def dtype(self):
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.embeddings.patch_embedding.weight.device

    def _rope_cos_sin(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rotary_pos_emb is not None
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        cos_table, sin_table = self.rotary_pos_emb.get_cos_sin(max_grid)
        indices = [
            torch.stack(
                (torch.arange(h).repeat_interleave(w), torch.arange(w).repeat(h)),
                dim=-1,
            ).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        indices = torch.cat(indices).to(cos_table.device)
        return cos_table[indices].flatten(1), sin_table[indices].flatten(1)

    def _encode(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        """Patch embedding plus ViT layers over the packed patch sequence."""
        config = self.config
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        if sum(lengths) != pixel_values.shape[0] or not lengths or min(lengths) <= 0:
            raise ValueError(
                "HYV4 pixel_values length does not match positive grid_thw"
            )
        hidden = self.embeddings(
            self.input_norm(pixel_values.to(device=self.device), self.dtype), grid_thw
        )
        cu = np.cumsum([0, *lengths], dtype=np.int32)
        max_seq = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu),
            dtype=torch.int32,
        )
        seq_lens = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu, self.device
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend, cu, config.hidden_size, self.tp_size, self.device
        )
        cos = sin = None
        if self.rotary_pos_emb is not None:
            cos, sin = self._rope_cos_sin(grid_thw)
        for layer in self.layers:
            hidden = layer(
                hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seq,
                cos=cos,
                sin=sin,
                sequence_lengths=seq_lens,
            )
        return hidden

    def _iter_item_states(
        self, hidden: torch.Tensor, grid_thw: list[list[int]]
    ) -> Iterable[torch.Tensor]:
        """Per-item `(frames, h, w, hidden)` slices with temporal padding."""
        config = self.config
        offset = 0
        for t, h, w in grid_thw:
            states = hidden[:, offset : offset + t * h * w].reshape(t, h, w, -1)
            offset += t * h * w
            if t > 1 and t % config.temporal_patch_size:
                pad = config.temporal_patch_size - t % config.temporal_patch_size
                states = torch.cat((states, states[-1:].expand(pad, -1, -1, -1)), dim=0)
            yield states

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> list[torch.Tensor]:
        hidden = self._encode(pixel_values, grid_thw)
        return [
            self.perceive(states) for states in self._iter_item_states(hidden, grid_thw)
        ]

    def prepare_encoder_cudagraph_metadata(
        self,
        grid_thw: list[list[int]],
        *,
        device: torch.device,
        max_seqlen_override: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Precompute the packed-forward inputs that depend only on grids."""
        grid_thw = [[int(v) for v in grid] for grid in grid_thw]
        metadata: dict[str, torch.Tensor | None] = {}
        metadata["pos_embeds"] = self.embeddings.compute_pos_embeds(
            grid_thw, self.dtype
        ).to(device)
        if self.rotary_pos_emb is not None:
            cos, sin = self._rope_cos_sin(grid_thw)
            metadata["rope_cos"] = cos.to(device)
            metadata["rope_sin"] = sin.to(device)
        else:
            metadata["rope_cos"] = metadata["rope_sin"] = None
        lengths = [h * w for t, h, w in grid_thw for _ in range(t)]
        cu = np.cumsum([0, *lengths], dtype=np.int32)
        max_seqlen = (
            max_seqlen_override
            if max_seqlen_override is not None
            else MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu)
        )
        metadata["max_seqlen"] = torch.tensor(max_seqlen, dtype=torch.int32)
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend, cu, self.config.hidden_size, self.tp_size, device
        )
        return metadata

    def forward_packed(
        self,
        patches: torch.Tensor,
        pos_embeds: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shape packed ViT forward for encoder CUDA graph capture/replay.

        `patches` holds the *embedded* patches (input norm + patch conv
        applied eagerly at replay time): both conv2d kernels (patch embed and
        merger) are cudnn calls whose results are not invariant to input
        shape (measured on GB200), so they cannot run on padded buffers
        without changing results bit-wise. Everything left in the graph —
        pos-embed add, per-layer LayerNorm/GEMM/RoPE/FlashAttn-varlen — is
        row-invariant and verified bitwise-identical to the eager path.

        All grid-dependent host logic must be precomputed via
        `prepare_encoder_cudagraph_metadata`. Returns the padded packed
        hidden states `(1, num_patch_slots, hidden_size)`; the per-item
        merger then runs eagerly on the real rows (in
        `postprocess_encoder_output`).
        """
        hidden = patches.unsqueeze(0) + pos_embeds.unsqueeze(0)
        for layer in self.layers:
            hidden = layer(
                hidden,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=rope_cos,
                sin=rope_sin,
                sequence_lengths=None,
            )
        return hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            for projection, shard in (
                ("q_proj", "q"),
                ("k_proj", "k"),
                ("v_proj", "v"),
            ):
                if f".{projection}." in name:
                    name = name.replace(f".{projection}.", ".qkv.")
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


def run_dp_sharded_hyv4_vision_tower(
    visual: HYV4VisionTransformer,
    pixel_values: torch.Tensor,
    grid_thw: list[list[int]],
    modality: str,
) -> tuple[torch.Tensor, ...]:
    """Run the HYV4 vision tower with whole media items sharded across the
    TP group (`--mm-encoder-tp-mode data`).

    Every rank holds the full tower weights and receives the full batch.
    Items are assigned to ranks by patch count, each rank encodes only its
    share, and per-item embeddings are exchanged with one padded all-gather
    and returned in the original item order on every rank.

    Unlike `run_dp_sharded_mrope_vision_model`, per-item output rows come
    from `vision_token_count`: the merger adds newline/extra tokens and
    pools video frames, so rows differ from `prod(grid) // merge**2`.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    patches_per_item = [t * h * w for t, h, w in grid_thw]
    cum_patches = [0, *itertools.accumulate(patches_per_item)]
    (item_to_rank, items_per_rank, _) = get_load_balance_assignment(
        patches_per_item, tp_size
    )
    cum_items_per_rank = [0, *itertools.accumulate(items_per_rank)]

    def rank_items(rank: int) -> list[int]:
        return item_to_rank[cum_items_per_rank[rank] : cum_items_per_rank[rank + 1]]

    # All-gather needs one shape on every rank; pad each rank's packed
    # output to the largest per-rank row count (computable locally).
    rows_per_item = [
        vision_token_count(grid, visual.config, modality) for grid in grid_thw
    ]
    max_rows = max(sum(rows_per_item[i] for i in rank_items(r)) for r in range(tp_size))

    local_items = rank_items(tp_rank)
    if local_items:
        pixels_local = torch.cat(
            [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in local_items]
        )
        embeds_local = torch.cat(
            visual(pixels_local, [grid_thw[i] for i in local_items]), dim=0
        )
    else:
        # Empty ranks all-gather placeholders in the tower's parameter dtype
        # (`visual.dtype`, the patch-embedding weight dtype for HYV4). HY4's
        # tower is homogeneous-dtype, so this is also its output dtype; a
        # tower with split dtypes would need its merger's weight dtype here
        # instead (cf. DeepSeek-VL's aligner.w2.weight.dtype).
        embeds_local = torch.empty(
            (0, visual.config.out_hidden_size),
            device=visual.device,
            dtype=visual.dtype,
        )
    if embeds_local.shape[0] < max_rows:
        embeds_local = torch.cat(
            (
                embeds_local,
                embeds_local.new_empty(
                    (max_rows - embeds_local.shape[0], embeds_local.shape[1])
                ),
            ),
            dim=0,
        )

    gathered = tensor_model_parallel_all_gather(embeds_local.contiguous(), dim=0)

    out: list[torch.Tensor | None] = [None] * len(grid_thw)
    for rank in range(tp_size):
        offset = rank * max_rows
        for i in rank_items(rank):
            out[i] = gathered[offset : offset + rows_per_item[i]]
            offset += rows_per_item[i]
    assert all(embed is not None for embed in out), "Found unassigned embeddings"
    return tuple(out)  # type: ignore[arg-type]
