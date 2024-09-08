from dataclasses import dataclass
from functools import partial
import itertools
import collections
import math
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union, Callable, Dict, Any)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tv
from PIL import Image

from vllm.attention import Attention, AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from .interfaces import SupportsMultiModal
from .llama import LlamaAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)

from vllm.transformers_utils.multimodal_processors.llamavl import LlamaVLImageProcessor

logger = init_logger(__name__)
MP_SCALE = 8
IMAGE_RES = 224

class LlamaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size, max_num_image, max_num_chunk, num_channels, height, width)`"""
    aspect_ratios: torch.Tensor
    """Shape: `(batch_size, max_num_image, 2)`"""
    num_chunks: List[List[int]]

# TODO: support LlamaImageEmbeddingInputs

LlavaImageInputs = LlamaImagePixelInputs
image_processor = None

def input_processor_for_llamavl(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("encoder_multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs
    global image_processor
    if image_processor is None:
        image_processor = LlamaVLImageProcessor(ctx.model_config.model)
    
    processed_image = image_processor(multi_modal_data["image"])
    llm_inputs["encoder_multi_modal_data"]["image"] = processed_image

    num_chunks = int(processed_image["aspect_ratios"].sum())
    assert ctx.model_config.hf_config.vision_chunk_size % 14 == 0, "chunk size should be multiple of 14"
    token_per_chunk = (ctx.model_config.hf_config.vision_chunk_size // 14) ** 2 + 1
    num_tokens = num_chunks * token_per_chunk
    llm_inputs["encoder_prompt"] = "<|image|>" * num_tokens
    llm_inputs["encoder_prompt_token_ids"] = [128256] * num_tokens

    assert "decoder_multi_modal_data" not in llm_inputs, "multi-modal data should be put in encoder message of LLaMA Vision"

    return llm_inputs

def get_max_llama_image_tokens(ctx: InputContext) -> int:
    logger.warning("need further check on max llama image tokens")
    return 1025 * 2


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

def resize_local_position_embedding(orig_pos_embed, grid_size):
    """
    Resize position embedding for vision encoder.
    Original position embedding is [n_tiles * n_tiles + 1, dim]
    New position embedding will be [grid_size[0] * grid_size[1] + 1, dim]
    """
    new_grid_size = to_2tuple(grid_size)
    orig_grid_size = to_2tuple(int(math.sqrt(len(orig_pos_embed) - 1)))
    new_seq_len = new_grid_size[0] * new_grid_size[1] + 1

    new_pos_emb_tok, new_pos_emb_img = (
        orig_pos_embed[:1],
        orig_pos_embed[1:],
    )
    logger.info(
        f"resizing position embedding grid-size from {orig_grid_size} to {new_grid_size}"
    )

    new_pos_emb_img = new_pos_emb_img.reshape(
        1, orig_grid_size[0], orig_grid_size[1], -1
    ).permute(0, 3, 1, 2)

    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1).reshape(
        1, new_grid_size[0] * new_grid_size[1], -1
    )[0]
    new_pos_embed = torch.cat([new_pos_emb_tok, new_pos_emb_img], dim=0)
    return new_pos_embed


def initialize_global_position_embedding_from_local(
    pos_and_cls_embed, grid_size, x_scale, y_scale
):
    """
    Takes a local position embedding for vision encoder and uses it
    to initialize the global position embedding.
    Input: local position embedding of shape [grid_size[0] * grid_size[1] + 1, dim]
    Returns: global position embedding of shape [x_scale, y_scale, grid_size[0] * grid_size[1] + 1, dim]
    Here x_scale and y_scale are the number of tiles along x-axis and y-axis respectively.
    """
    pos_embed = pos_and_cls_embed[1:]
    cls_embed = pos_and_cls_embed[0].view(1, 1, 1, -1)
    grid_size = to_2tuple(grid_size)
    new_pos_emb_img = pos_embed.reshape(1, grid_size[0], grid_size[1], -1).permute(
        0, 3, 1, 2
    )
    new_grid_size = (x_scale * grid_size[0], y_scale * grid_size[1])
    new_pos_emb_img = F.interpolate(
        new_pos_emb_img,
        size=new_grid_size,
        mode="bilinear",
        align_corners=True,
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 3, 1)
    new_pos_emb_img = new_pos_emb_img.view(
        x_scale, grid_size[0], y_scale, grid_size[1], -1
    )
    new_pos_emb_img = new_pos_emb_img.permute(0, 2, 1, 3, 4).contiguous()
    new_pos_emb_img = new_pos_emb_img.reshape(
        x_scale, y_scale, grid_size[0] * grid_size[1], -1
    )
    cls_embed = cls_embed.expand(x_scale, y_scale, -1, -1)
    pos_and_cls_embed = torch.cat([cls_embed, new_pos_emb_img], dim=2)
    return pos_and_cls_embed


def resize_global_position_embedding(pos_and_cls_embed, grid_size, x_scale, y_scale):
    """
    Takes a global position embedding for vision encoder and resizes it to new size.
    Input: global position embedding of shape [x_old, y_old, old_grid_size[0] * old_grid_size[1] + 1, dim]
    Returns: global position embedding of shape [x_scale, y_scale, grid_size[0] * grid_size[1] + 1, dim]
    Here x_scale and y_scale are the number of tiles along x-axis and y-axis respectively.
    """
    # first remove cls token
    pos_embed = pos_and_cls_embed[:, :, 1:]
    cls_embed = pos_and_cls_embed[:, :, 0].unsqueeze(2)

    xs_old, ys_old, ntok, dim = pos_embed.shape
    old_grid_size = int(math.sqrt(ntok))

    # move to correct form for interpolation
    pos_embed = pos_embed.view(xs_old, ys_old, old_grid_size, old_grid_size, dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(xs_old * old_grid_size, ys_old * old_grid_size, dim)
    pos_embed = pos_embed.unsqueeze(0)

    # interpolate
    new_size = (grid_size[0] * x_scale, grid_size[1] * y_scale)
    pos_embed = pos_embed.permute(0, 3, 1, 2)
    pos_embed_resized = F.interpolate(
        pos_embed,
        size=new_size,
        mode="bilinear",
        align_corners=True,
    )
    pos_embed = pos_embed_resized.permute(0, 2, 3, 1)[0]

    # move it back in place
    pos_embed = pos_embed.view(x_scale, grid_size[0], y_scale, grid_size[1], dim)
    pos_embed = pos_embed.permute(0, 2, 1, 3, 4).contiguous()
    pos_embed = pos_embed.view(x_scale, y_scale, grid_size[0] * grid_size[1], dim)

    # interpolate cls token
    cls_embed = cls_embed.permute(2, 3, 0, 1)
    cls_embed_resized = F.interpolate(
        cls_embed,
        size=(x_scale, y_scale),
        mode="bilinear",
        align_corners=True,
    )
    cls_embed = cls_embed_resized.permute(2, 3, 0, 1)
    # add cls token back in
    pos_and_cls_embed = torch.cat([cls_embed, pos_embed], dim=2)

    return pos_and_cls_embed


def build_encoder_attention_mask(
    x: torch.Tensor,
    ar: torch.Tensor,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """
    Build vision encoder attention mask that omits padding tokens.
    """
    masks = []
    for arx in ar:
        mask_i = torch.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[: arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.view(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * torch.finfo(x.dtype).min
        mask_i = mask_i.unsqueeze(0)
        masks.append(mask_i)
    masks = torch.stack(masks).to(x.device).expand(-1, n_heads, -1, -1)
    return masks


def expand_num_tokens_to_mult8(x):
    num_pad_tokens = 8 - (x.shape[-2] % 8)
    if num_pad_tokens == 0:
        return x, 0
    else:
        return (
            torch.cat(
                [
                    x,
                    torch.zeros(
                        (x.shape[0], x.shape[1], num_pad_tokens, x.shape[-1]),
                        dtype=x.dtype,
                        device=x.device,
                    ),
                ],
                dim=-2,
            ),
            num_pad_tokens,
        )


def contract_num_tokens_from_mult8(x, num_pad_tokens):
    if num_pad_tokens == 0:
        return x
    return x[:, :, :-num_pad_tokens]

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def _get_full_row_masked_out_mask(
    attn_bias,
    negative_inf_value,
):
    """
    attn_bias should be a 4D tensor of shape [B, H, S1, S2]
    where B is the batch size, H is the number of heads,
    and S1/S2 are the sequence lengths. This returns
    a 4D tensor of shape [B, H, S1, 1] which stores boolean
    values which are 0 if the a full row in the last dimension
    contains negative infinity values, otherwise it's 1.
    """
    return (attn_bias != negative_inf_value).any(dim=-1).type_as(attn_bias)[..., None]

# Image encoder for inference
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class ColumnParallelConv2dPatch(torch.nn.Module):
    """Conv2D Patching layer with model parallelism.
    Column parallel over unfolded input.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        bias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride)
        # self._linear = ColumnParallelLinear(
        #     in_channels * kernel_size[0] * kernel_size[1],
        #     out_channels,
        #     bias=bias,
        # )
        self._linear = nn.Linear(
            in_channels * kernel_size[0] * kernel_size[1],
            out_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        x = self._linear(x)
        # x = F.linear(x, self._linear.weight)
        # x = gather_from_tensor_model_parallel_region(x)
        return x


class ImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        # layers
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        # self.c_fc = ColumnParallelLinear(
        #     dim,
        #     hidden_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)
        # self.c_proj = RowParallelLinear(
        #     hidden_dim,
        #     dim,
        #     bias=True,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )
        self.non_linearity = act_layer()
        self.dropout = dropout

    def forward(self, x):
        hidden = self.c_fc(x)
        # hidden = F.linear(x, self.c_fc.weight, self.c_fc.bias)
        hidden = self.non_linearity(hidden)
        hidden = self.c_proj(hidden)
        # hidden = F.linear(hidden, self.c_proj.weight)
        # hidden = reduce_from_tensor_model_parallel_region(hidden)
        # hidden += self.c_proj.bias
        return hidden


class ImageAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        n_heads,
    ):
        super().__init__()
        model_parallel_size = get_tensor_model_parallel_world_size()
        qkvo_replication = 1
        if model_parallel_size > 16:
            qkvo_replication = model_parallel_size // 8

        self.n_kv_heads = n_heads
        self.n_local_heads = n_heads * qkvo_replication // model_parallel_size
        self.n_local_kv_heads = (
            self.n_kv_heads * qkvo_replication // model_parallel_size
        )
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        # The model provided by llama is with bias=True, but the weight does not contain bias
        # During runtime, the llama executor set bias to zero. We use bias=False here to match the behavior
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        # self.wq = ColumnParallelLinear(
        #     dim,
        #     qkvo_replication * n_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     dim,
        #     qkvo_replication * self.n_kv_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     dim,
        #     qkvo_replication * self.n_kv_heads * self.head_dim,
        #     bias=True,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     qkvo_replication * n_heads * self.head_dim,
        #     dim,
        #     bias=True,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )
        self.qkvo_replication = qkvo_replication

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        xq, xk, xv = [
            F.linear(x, w, b)
            for (w, b) in [
                (self.wq.weight, self.wq.bias),
                (self.wk.weight, self.wk.bias),
                (self.wv.weight, self.wv.bias),
            ]
        ]

        bs, slen, _ = xq.shape

        xq = xq.view(bs, slen, self.n_local_heads, self.head_dim)
        xk = xk.view(bs, xk.shape[1], self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs, xv.shape[1], self.n_local_kv_heads, self.head_dim)

        xq, xk, xv = [tensor.transpose(1, 2) for tensor in (xq, xk, xv)]

        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bs, slen, -1)

        out = F.linear(attn_output, self.wo.weight)
        # out = reduce_from_tensor_model_parallel_region(out)
        out = out / self.qkvo_replication
        # out += self.wo.bias
        return out


class ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.head_dim = d_model // self.n_heads
        self.attn = ImageAttention(
            dim=d_model,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = ImageFeedForward(
            dim=d_model,
            hidden_dim=int(mlp_ratio * d_model),
            dropout=0.0,
            act_layer=act_layer,
        )
        self.ln_2 = LayerNorm(d_model)
        self.gated = gated
        if gated:
            self.gate_attn = nn.Parameter(torch.zeros(1))
            self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        _gate_attn = 1 if not self.gated else self.gate_attn.tanh()
        _gate_ffn = 1 if not self.gated else self.gate_ffn.tanh()
        x = x + _gate_attn * self.attn(self.ln_1(x), mask=mask)
        x = x + _gate_ffn * self.mlp(self.ln_2(x))
        return x


class ImageTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        gated: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ImageTransformerBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    gated=gated,
                )
                for _ in range(self.layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_intermediate=None, mask=None):
        out = []
        for idx, r in enumerate(self.resblocks):
            if return_intermediate is not None and idx in return_intermediate:
                out.append(x)
            x = r(x, mask=mask)
        if return_intermediate is not None:
            return x, torch.stack(out, dim=-1)
        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        max_num_tiles: int,
        # ckpt_path: str = None,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1280,
        layers: int = 32,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        in_channels: int = 3,
        # load_ckpt: bool = False,
        n_global_layers: int = 2,
        global_model: bool = False,
        return_intermediate=None,
    ):
        super().__init__()
        self.global_model = global_model
        self.return_intermediate = return_intermediate
        self.max_num_tiles = max_num_tiles
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.conv1 = ColumnParallelConv2dPatch(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_post = LayerNorm(width)
        self.ln_pre = LayerNorm(width)
        self.transformer = ImageTransformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer
        )
        # pre and post tile position embedding
        self.global_transformer = ImageTransformer(
            width, n_global_layers, heads, mlp_ratio, act_layer=act_layer, gated=True
        )
        # pre and post tile position embedding
        self.pre_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.post_tile_pos_embed = TilePositionEmbedding(
            num_tiles=max_num_tiles,
            width=width,
            gated=True,
        )
        self.gated_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.grid_size[0] * self.grid_size[1] + 1,
                width,
            )
        )
        self.gated_positional_embedding_gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool = True,
        missing_keys: List[str] = None,
        unexpected_keys: List[str] = None,
        error_msgs: List[str] = None,
        return_state_dict: bool = False,
    ) -> None:
        orig_pos_embed = state_dict.get(prefix + "positional_embedding")
        if orig_pos_embed is not None:
            new_pos_embed = resize_local_position_embedding(
                orig_pos_embed, self.grid_size
            )
            state_dict[prefix + "positional_embedding"] = new_pos_embed
        if hasattr(self, "gated_positional_embedding"):
            if prefix + "gated_positional_embedding" not in state_dict:
                # resize positional_embedding to fit the new grid size
                global_pos_embed = initialize_global_position_embedding_from_local(
                    new_pos_embed,
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
                state_dict[prefix + "gated_positional_embedding_gate"] = torch.zeros(
                    1, dtype=global_pos_embed.dtype
                )
                logger.info(
                    f"Initialized global positional embedding with size {global_pos_embed.size()}"
                )
            else:
                global_pos_embed = resize_global_position_embedding(
                    state_dict[prefix + "gated_positional_embedding"],
                    self.grid_size,
                    self.max_num_tiles,
                    self.max_num_tiles,
                )
                logger.info(
                    f"Resized global positional embedding from {state_dict[prefix + 'gated_positional_embedding'].size()} to {global_pos_embed.size()}"
                )
                state_dict[prefix + "gated_positional_embedding"] = global_pos_embed
        if return_state_dict:
            return state_dict

    def apply_positional_embedding(self, x, ar):
        out = []
        # apply regular position embedding
        bsz, num_chunks, num_tokens, dim = x.shape
        x = x.view(bsz * num_chunks, num_tokens, dim)
        x = x + self.positional_embedding * (
            1 - self.gated_positional_embedding_gate.tanh()
        )
        x = x.view(bsz, num_chunks, num_tokens, dim)
        for idx, arx in enumerate(ar):
            _pos_embed = self.gated_positional_embedding[: arx[0], : arx[1]]
            _pos_embed = _pos_embed.reshape(arx[0] * arx[1], *_pos_embed.shape[2:])
            x[idx, : arx[0] * arx[1]] += (
                _pos_embed * self.gated_positional_embedding_gate.tanh()
            )
        return x

    def apply_class_embedding(self, x):
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        return x

    def forward(self, images: torch.Tensor, ar: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            num_concurrent_media = 1
            bsz, num_chunks, nch, w, h = images.shape
        else:
            bsz, num_concurrent_media, num_chunks, nch, w, h = images.shape

        images = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        ar = ar.reshape(bsz * num_concurrent_media, 2)

        # patch embedding
        x = images.reshape(bsz * num_concurrent_media * num_chunks, nch, w, h)
        x = self.conv1(x)
        _, ntok, dim = x.shape
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)

        # tile embeddings
        x = self.pre_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media * num_chunks, ntok, dim)

        # apply cls token
        x = self.apply_class_embedding(x)
        ntok += 1

        # apply position embeddings
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok, dim)
        x = self.apply_positional_embedding(x, ar)

        x = self.ln_pre(x)
        npad, attn_mask = 0, None
        x, npad = expand_num_tokens_to_mult8(x)
        attn_mask = build_encoder_attention_mask(x, ar, ntok, num_chunks, 1)
        x = x.view(bsz * num_concurrent_media, -1, dim)
        x, int_x = self.transformer(
            x, return_intermediate=self.return_intermediate, mask=attn_mask
        )

        x = self.ln_post(x)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = self.post_tile_pos_embed(x, ar)
        x = x.reshape(bsz * num_concurrent_media, num_chunks * (ntok + npad), dim)
        x = self.global_transformer(x, mask=attn_mask)
        x = x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, dim)
        x = contract_num_tokens_from_mult8(x, npad)

        # adding back intermediate layer outputs
        x = x.reshape(bsz, num_concurrent_media, num_chunks, ntok, dim)
        int_x = int_x.reshape(bsz * num_concurrent_media, num_chunks, ntok + npad, -1)
        int_x = contract_num_tokens_from_mult8(int_x, npad)
        int_x = int_x.reshape(bsz, num_concurrent_media, num_chunks, ntok, -1)
        x = torch.cat([x, int_x], dim=-1)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self._register_load_state_dict_pre_hook(self.load_hook)

    def forward(self, x):
        x1, x3 = [F.linear(x, w) for w in [self.w1.weight, self.w3.weight]]
        x1 = F.silu(x1)
        x_in = x1 * x3
        out = F.linear(x_in, self.w2.weight)
        # out = reduce_from_tensor_model_parallel_region(out)
        return out

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "mlp.fc1_weight" in state_dict:
            fc1_weight, fc3_weight = state_dict.pop(prefix + "mlp.fc1_weight").chunk(2)
            state_dict[prefix + "w1.weight"] = fc1_weight
            state_dict[prefix + "w3.weight"] = fc3_weight

        if prefix + "mlp.fc2_weight" in state_dict:
            fc2_weight = state_dict.pop(prefix + "mlp.fc2_weight")
            state_dict[prefix + "w2.weight"] = fc2_weight

class LlamaVLAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(self.load_hook)


    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "wqkv.weight" in state_dict:
            state_dict[prefix + "qkv_proj.weight"] = state_dict.pop(prefix + "wqkv.weight")
        if prefix + "wo.weight" in state_dict:
            state_dict[prefix + "o_proj.weight"] = state_dict.pop(prefix + "wo.weight")

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args, cache_config: Optional[CacheConfig] = None):
        """
        Initialize a TransformerBlock.
        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        # TODO: remove "use_scaled_rope" from args
        self.attention = LlamaVLAttention(
            config=args,
            hidden_size=args.dim,
            num_heads=self.n_heads,
            num_kv_heads=args.n_kv_heads,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
            max_position_embeddings=512,
            quant_config=None,
            bias=False,
            cache_config=cache_config,
            prefix=f"tb.{layer_id}.self_attn",
        )
        # logger.warning("skip attention")
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(
                prefix + "feed_forward.mlp.layer_norm_weight"
            )
        if prefix + "attention.wqkv.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(
                prefix + "attention.wqkv.layer_norm_weight"
            )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.LongTensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the TransformerBlock.
        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.
        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
        """
        # TODO: need to compute qkv and then do attention
        h = self.attention.forward(
            positions=positions,
            hidden_states=self.attention_norm(x),
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        h = h + x
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        width: int,
        gated: bool = False,
    ):
        super().__init__()
        self.num_tiles = num_tiles
        self.width = width
        self.embedding = nn.Parameter(
            torch.randn(num_tiles, num_tiles, 1, width) / math.sqrt(width)
        )
        self.gated = gated
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # load the weights from the checkpoint
        embed = state_dict.get(prefix + "embedding")
        if embed is not None:
            # reshape the weights to the correct shape
            nt_old, nt_old, _, w = embed.shape
            logger.info(
                f"Resizing tile embedding from {nt_old}x{nt_old} to {self.num_tiles}x{self.num_tiles}"
            )
            embed_new = TilePositionEmbedding._dynamic_resize(embed, self.num_tiles)
            # assign the weights to the module
            state_dict[prefix + "embedding"] = embed_new

    @staticmethod
    def _dynamic_resize(embed: torch.Tensor, num_tiles: int):
        nt_old, nt_old, _, w = embed.shape
        embed = embed.permute(2, 3, 0, 1)

        embed_new = F.interpolate(
            embed,
            size=(num_tiles, num_tiles),
            mode="bilinear",
            align_corners=True,
        )
        # reshape the weights to the correct shape
        embed_new = embed_new.permute(2, 3, 0, 1)
        return embed_new

    def forward(self, x: torch.Tensor, ar: torch.Tensor, num_tiles: int = None):
        embed = self.embedding
        if num_tiles is None:
            num_tiles = self.num_tiles
        elif num_tiles > self.num_tiles:
            embed = TilePositionEmbedding._dynamic_resize(self.embedding, num_tiles)
        out_pos_embed = torch.zeros(
            x.shape[0], num_tiles, 1, self.width, device=x.device, dtype=x.dtype
        )
        for idx, arx in enumerate(ar):
            w, h = arx
            out_pos_embed[idx, : w * h] = embed[:w, :h].reshape(w * h, 1, self.width)
        if self.gated:
            out_pos_embed = out_pos_embed * self.gate.tanh()
        x = x + out_pos_embed
        return x


def _noinit(x):
    return x


class CrossAttention(torch.nn.Module):
    """Cross attention layer with model-parallel attention layers."""

    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        replication_factor = 1
        if self.model_parallel_size > 8:
            replication_factor = self.model_parallel_size // MP_SCALE
        n_kv_heads *= replication_factor

        assert n_heads % n_kv_heads == 0


        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        # self.wq = ColumnParallelLinear(
        #     dim,
        #     n_heads * head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=_noinit,
        # )

        # self.wk = ColumnParallelLinear(
        #     dim,
        #     n_kv_heads * head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=_noinit,
        # )
        # self.wv = ColumnParallelLinear(
        #     dim,
        #     n_kv_heads * head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=_noinit,
        # )
        # self.wo = RowParallelLinear(
        #     n_heads * head_dim,
        #     dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=_noinit,
        # )

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=norm_eps,
        )

        # cross-attention heads are model parallel similar to
        # self-attention, and we also use the identical KV head
        # combination to ensure parity with the corresponding
        # trunk LLM (i.e., group query attention) -- @dubeya
        # local heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.n_heads % self.model_parallel_size == 0
        assert self.n_kv_heads % self.model_parallel_size == 0
        self.n_local_heads = self.n_heads // self.model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "inner_attention.q_norm.weight" in state_dict:
            q_weight = state_dict.pop(prefix + "inner_attention.q_norm.weight")
            state_dict[prefix + "q_norm.weight"] = q_weight
        if prefix + "inner_attention.k_norm.weight" in state_dict:
            k_weight = state_dict.pop(prefix + "inner_attention.k_norm.weight")
            state_dict[prefix + "k_norm.weight"] = k_weight
        if prefix + "wkv.weight" in state_dict:
            wk, wv = state_dict.pop(prefix + "wkv.weight").chunk(2)
            state_dict[prefix + "wk.weight"] = wk
            state_dict[prefix + "wv.weight"] = wv

    def _compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        bsz = xattn_tokens.shape[0]
        xk = self.wk(xattn_tokens)
        xv = self.wv(xattn_tokens)

        _, seqlen_y, _ = xk.shape

        xk = xk.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)

        xk, xv = [tensor.transpose(1, 2) for tensor in (xk, xv)]

        # repeat k/v heads if n_kv_heads < n_heads
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)

        xk = self.k_norm(xk)

        return torch.stack([xk, xv])

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self._compute_xattn_kv_cache(xattn_tokens)

    def unpack_value(self, x: torch.Tensor, positions: torch.LongTensor, attn_metadata: AttentionMetadata, xattn_mask: torch.Tensor, full_text_row_masked_out_mask: torch.Tensor):
        x_unpacked = torch.zeros(attn_metadata.num_prefills, attn_metadata.max_query_len, x.shape[-1], device=x.device, dtype=x.dtype)
        positions_unpacked = torch.zeros(attn_metadata.num_prefills, attn_metadata.max_query_len, device=positions.device, dtype=positions.dtype)
        xattn_mask = xattn_mask[:, :, :attn_metadata.max_query_len]
        # position
        start_pos = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens_tensor):
            end_pos = start_pos + seq_len
            x_unpacked[i, :seq_len] = x[start_pos:end_pos]
            positions_unpacked[i, :seq_len] = positions[start_pos:end_pos]
            xattn_mask[i, 0, seq_len:] = torch.finfo(xattn_mask.dtype).min
            start_pos = end_pos
        # xattn_mask = xattn_mask[:, :, :attn_metadata.max_query_len]
        # full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, :attn_metadata.max_query_len]
        return x_unpacked, positions_unpacked, xattn_mask, full_text_row_masked_out_mask

    def pack_value(self, x:torch.Tensor, attn_metadata: AttentionMetadata):
        x_packed = torch.zeros(attn_metadata.num_prefill_tokens, x.shape[-1], device=x.device, dtype=x.dtype)
        start_pos = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens_tensor):
            end_pos = start_pos + seq_len
            x_packed[start_pos:end_pos] = x[i, :seq_len]
            start_pos = end_pos
        return x_packed

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_cache: torch.Tensor,
        positions: torch.LongTensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        xq = F.linear(x, self.wq.weight)
        n_token = xq.shape[0]
        xq, positions, xattn_mask, full_text_row_masked_out_mask = self.unpack_value(xq, positions, attn_metadata, xattn_mask, full_text_row_masked_out_mask)
        bsz, seqlen, _ = xq.shape

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = self.q_norm(xq)
        xq = xq.transpose(1, 2) # [bs, n_head, seq_len, head_dim]

        xk, xv = xattn_cache

        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=xattn_mask, dropout_p=0.0
        )
        
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1).contiguous()
        output = self.pack_value(output, attn_metadata)

        output = output * full_text_row_masked_out_mask
        out = F.linear(output, self.wo.weight)
        return out


class CrossAttentionTransformerBlock(torch.nn.Module):
    """Cross-attention transformer block with tanh-gated attention and feedforward."""

    def __init__(
        self,
        args,
        layer_id: int,
        no_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = CrossAttention(
            dim=args.dim,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=args.norm_eps,
        )

        self.attention_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_attn = torch.nn.Parameter(torch.zeros(1))

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            multiple_of=args.multiple_of,
        )
        self.ffn_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
        )
        self.gate_ffwd = torch.nn.Parameter(torch.zeros(1))

        logger.warning("todo put hook in correct place")
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.no_ffn = no_ffn

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "gate_attn" in state_dict:
            attn_gate = state_dict.pop(prefix + "gate_attn")
            if attn_gate.dim() == 1:
                attn_gate = attn_gate[0].view(1)
            if attn_gate.dim() == 3:
                attn_gate = attn_gate.view(1)
            state_dict[prefix + "gate_attn"] = attn_gate
        if prefix + "gate_ffwd" in state_dict:
            ffn_gate = state_dict.pop(prefix + "gate_ffwd")
            if ffn_gate.dim() == 1:
                ffn_gate = ffn_gate[0].view(1)
            if ffn_gate.dim() == 3:
                ffn_gate = ffn_gate.view(1)
            state_dict[prefix + "gate_ffwd"] = ffn_gate
        if prefix + "feed_forward.mlp.layer_norm_weight" in state_dict:
            state_dict[prefix + "ffn_norm.weight"] = state_dict.pop(
                prefix + "feed_forward.mlp.layer_norm_weight"
            )
        if prefix + "attention.wq.layer_norm_weight" in state_dict:
            state_dict[prefix + "attention_norm.weight"] = state_dict.pop(
                prefix + "attention.wq.layer_norm_weight"
            )

    def compute_xattn_kv_cache(self, xattn_tokens: torch.Tensor) -> torch.Tensor:
        return self.attention.compute_xattn_kv_cache(xattn_tokens)

    def forward(
        self,
        x: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_cache: torch.Tensor,
        positions: torch.LongTensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        _attn_out = self.attention(
            x=self.attention_norm(x),
            xattn_mask=xattn_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            xattn_cache=xattn_cache,
            positions=positions,
            attn_metadata=attn_metadata
        )
        h = x + self.gate_attn.tanh() * _attn_out
        _ffn = self.feed_forward(self.ffn_norm(h))
        _ffn = full_text_row_masked_out_mask * _ffn  # type: ignore
        h = h + self.gate_ffwd.tanh() * _ffn * float(not self.no_ffn)
        return h


class DummyCrossAttentionTransformerBlock:
    """Dummy cross-attention transformer block with tanh-gated attention and feedforward."""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class DummySelfAttentionTransformerBlock:
    """Dummy self-attention transformer block"""

    def __call__(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return x


class CrossAttentionTransformerVision(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        return_intermediate = "3,7,15,23,30"
        self.vision_input_dim = 1280
        self.image_res = args.vision_chunk_size
        self.max_num_chunks = args.vision_max_num_chunks
        if return_intermediate is not None:
            return_intermediate = [int(l) for l in return_intermediate.split(",")]
            self.vision_input_dim = (
                len(return_intermediate) + 1
            ) * self.vision_input_dim
        self.patch_size = 14
        self.vision_encoder = VisionEncoder(
            max_num_tiles=4,
            image_size=args.vision_chunk_size,
            patch_size=self.patch_size,
            n_global_layers=8,
            global_model=True,
            return_intermediate=return_intermediate,
        )
        # vision token projection
        self.vision_projection = nn.Linear(
            self.vision_input_dim,
            args.dim,
            bias=True,
        )
        # self.vision_projection = ColumnParallelLinear(
        #     self.vision_input_dim,
        #     args.dim,
        #     bias=True,
        #     init_method=lambda x: x,
        # )

    def forward(
        self, images: torch.Tensor, aspect_ratios: torch.Tensor
    ) -> torch.Tensor:
        # vision_tokens: (B, T, D)
        # aspect_ratios: (B, T)
        # h: (B, T, D)
        vision_tokens = self.vision_encoder(
            images.to(dtype=torch.bfloat16), aspect_ratios
        )

        vision_tokens = F.linear(vision_tokens, self.vision_projection.weight)
        # vision_tokens = gather_from_tensor_model_parallel_region(vision_tokens)
        return vision_tokens


class CrossAttentionTransformerText(torch.nn.Module):
    INFERENCE_IMAGE_TOKEN_ID = 128010

    def __init__(self, args, cache_config:Optional[CacheConfig]) -> None:
        super().__init__()
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        assert args.vocab_size > 0
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads // self.model_parallel_size
        assert self.vocab_size % self.model_parallel_size == 0
        self.tok_embeddings = VocabParallelEmbedding(
            args.vocab_size, args.dim,
            padding_size=self.model_parallel_size,
        )
        self.pos_embeddings = None
        # final norm layer (not necessary for post-norm)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # output layer
        # self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # self.output = ColumnParallelLinear(
        #     args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        # )

        self.n_llama_layers = args.n_layers
        self.model_dim = args.dim

        # BLOCKS

        self.fusion_schedule = self._init_fusion_schedule(
            args.vision_num_cross_attention_layers
        )
        self.learnable_embedding = VocabParallelEmbedding(
            max(get_tensor_model_parallel_world_size(), 8),
            args.dim,
            padding_size=self.model_parallel_size,
        )
        self.num_frozen_embeddings = self.tok_embeddings.num_embeddings
        self._thresh = self.num_frozen_embeddings - 1

        # transformer blocks
        self.layers = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()
        for i in range(args.n_layers):
            layer_id = i
            block = TransformerBlock(args=args, layer_id=layer_id, cache_config=cache_config)
            self.layers.append(block)
            if layer_id in self.fusion_schedule:
                xa_layer_id = self.fusion_schedule.index(layer_id) + args.n_layers
                block = CrossAttentionTransformerBlock(
                    args,
                    layer_id=xa_layer_id,
                )
                self.cross_attention_layers.append(block)

        # add xattn and dummy layers to avoid conditionals in forward()
        self.text_and_xattn_layers = []

        for idx, layer in enumerate(self.layers):
            if idx in self.fusion_schedule:
                xattn_layer_idx = self.fusion_schedule.index(idx)
                xattn_layer = self.cross_attention_layers[xattn_layer_idx]
            else:
                xattn_layer_idx = 0
                xattn_layer = DummyCrossAttentionTransformerBlock()

            self.text_and_xattn_layers.append(
                (
                    layer,
                    xattn_layer,
                    xattn_layer_idx,
                )
            )
        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
        )

        self._register_load_state_dict_pre_hook(self.load_hook)

        self.args = args
        self.cache_is_setup = False
        self.max_seq_len = args.max_seq_len

    def _init_fusion_schedule(
        self,
        num_layers: int,
    ) -> List[int]:
        llama_layers = list(range(self.n_llama_layers))

        # uniformly spread the layers
        k = math.ceil(len(llama_layers) / num_layers)
        return llama_layers[::-1][::k][:num_layers][::-1]

    def get_partially_trainable_embedding(self, x):
        xz = torch.zeros_like(x, device=x.device)
        oz = torch.ones_like(x, device=x.device)
        x_orig = torch.minimum(x, torch.tensor(self._thresh, device=x.device))
        x_new = (
            torch.maximum(x, torch.tensor(self._thresh + 1, device=x.device))
            - self.num_frozen_embeddings
        )

        mask_orig = torch.where(x >= self.num_frozen_embeddings, xz, oz).unsqueeze(-1)
        mask_new = torch.where(x < self.num_frozen_embeddings, xz, oz).unsqueeze(-1)

        x_orig = self.tok_embeddings(x_orig)
        x_new = self.learnable_embedding(x_new).type_as(x_orig)
        return x_orig * mask_orig.type_as(x_orig) + x_new * mask_new.type_as(x_new)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if "rope.freqs" in state_dict:
            del state_dict["rope.freqs"]

    def forward(
        self,
        positions: torch.LongTensor,
        h: torch.Tensor,
        xattn_mask: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        xattn_caches: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        # assert self.cache_is_setup, "Please set up cache before calling forward"
        # mask = self.mask_cache.index_select(2, positions)
        # freqs_cis = self.freqs_cis.index_select(0, positions)

        for idx, (
            layer,
            xattn_layer,
            xattn_layer_idx,
        ) in enumerate(self.text_and_xattn_layers):
            # TODO: a hack now. skip decode cross attention
            if xattn_mask is not None:
                h = xattn_layer(
                    x=h,
                    xattn_mask=xattn_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    xattn_cache=xattn_caches[xattn_layer_idx],
                    positions=positions,
                    attn_metadata=attn_metadata,
                )
            h = layer(
                x=h,
                # mask=mask,
                # freqs_cis=freqs_cis,
                positions=positions,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
            )

        h = self.norm(h)
        return h
        # output = F.linear(h, self.output.weight)
        # output = gather_from_tensor_model_parallel_region(output)
        # return output.float()

    def _get_xattn_mask(
        self,
        num_tokens,
        text_device,
        text_dtype,
        vision_tokens,
        cross_attention_masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert vision_tokens is not None, "Vision tokens must be provided"
        vision_seqlen = vision_tokens.shape[3]
        assert (
            vision_tokens.shape[1] == cross_attention_masks.shape[2]
        ), f"Mismatch in number of images given and number of masks given {vision_tokens.shape} {cross_attention_masks.shape}"
        assert (
            vision_tokens.shape[2] == cross_attention_masks.shape[3]
        ), f"Vision tokens shape {vision_tokens.shape} mismatch with xattn shape {cross_attention_masks.shape}"
        assert (
            num_tokens == cross_attention_masks.shape[1]
        ), f"Mismatch in text sequence length and cross attention mask sequence length {num_tokens} {cross_attention_masks.shape}"
        _, _, _, num_image_tokens, image_token_dim = tuple(vision_tokens.shape)
        bsz, ntext, nimg, nchunks = cross_attention_masks.shape
        cross_attention_masks = (
            cross_attention_masks.repeat_interleave(vision_seqlen, dim=2)
            .view(bsz, ntext, -1)
            .unsqueeze(1)
        )
        full_text_row_masked_out_mask = _get_full_row_masked_out_mask(
            cross_attention_masks,
            torch.finfo(cross_attention_masks.dtype).min,
        )
        cross_attention_masks *= full_text_row_masked_out_mask

        return (
            cross_attention_masks.to(device=text_device, dtype=text_dtype),
            full_text_row_masked_out_mask,
        )


class VariableSizeImageTransform(object):
    """
    The variable size image transform will resize the image dynamically
    based on the image aspect ratio and the number of image chunks we allow.
    The algorithm will not upsample low-res images to fit a certain aspect
    ratio, because that leads to a significant degradation in image quality.
    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image chunks, it will find the closest aspect ratio that
    is allowed within 16 image chunks, i.e., 2:5 = 2 horizontal patches and
    5 vertical patches, giving a total of 10 chunks.
    The image will then be resized to products of the base size (default is
    224px because MetaCLIP takes that), so in this case it will  be resized to
    2*224:5*224 = 448:1120, where we maintain the original aspect ratio and
    pad with the mean value for the rest. This approach minimizes the amount
    of padding required for any arbitrary resolution.
    The final output will therefore be of shape (11, 3, 224, 224), where 10
    patches are coming from the resizing and chunking, and the first patch
    is a downsampled version of the image that preserves aspect ratios.
    """

    def __init__(self, size: int = IMAGE_RES) -> None:
        self.size = size
        self.to_tensor = tv.ToTensor()
        self._mean = (0.48145466, 0.4578275, 0.40821073)
        self._std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = tv.Normalize(
            mean=self._mean,
            std=self._std,
            inplace=True,
        )

    @staticmethod
    def _factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )

    def _find_supported_aspect_ratios(self, num_chunks: int):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks.
        For example, with `num_chunks=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(num_chunks, 0, -1):
            _factors = sorted(VariableSizeImageTransform._factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)
        return asp_dict

    def _find_closest_aspect_ratio(
        self, num_chunks: int, img_width: int, img_height: int
    ) -> Tuple:
        """
        Given an image width, height and target number of chunks
        this function will find the closest supported aspect ratio.
        """
        tgt_ar = img_width / img_height
        asp_dict = self._find_supported_aspect_ratios(num_chunks)
        cl_d, cl_p = 1e23, None
        if tgt_ar >= 1:
            cl_p = min(
                [k for k in asp_dict.keys() if k <= tgt_ar],
                key=lambda x: abs(x - tgt_ar),
            )
            v = asp_dict[cl_p]
            # select width
            widths = [(idx, self.size * vv[0]) for idx, vv in enumerate(v)]
            tgt_idx = max(widths, key=lambda x: x[1])[0]
        else:
            cl_p = min(
                [k for k in asp_dict.keys() if k > tgt_ar],
                key=lambda x: abs(1 / x - 1 / tgt_ar),
            )
            v = asp_dict[cl_p]
            # select height
            heights = [(idx, self.size * vv[1]) for idx, vv in enumerate(v)]
            tgt_idx = max(heights, key=lambda x: x[1])[0]
        out = v[tgt_idx]
        return out

    def _resize(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = target_width
            new_h = math.floor(new_w / scale)
        else:
            # height >= width
            new_h = target_height
            new_w = math.floor(new_h * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _resize_max_side_to_size(
        self,
        image: Image.Image,
    ) -> Image.Image:
        # Resize longer edge to given size.
        w, h = image.size
        scale = w / h

        if scale > 1.0:
            # width > height
            new_w = max(self.size, w)
            new_h = math.floor(new_w / scale)
        else:
            # height >= width
            new_h = max(self.size, h)
            new_w = math.floor(new_h * scale)

        image = F.resize(image, (new_h, new_w))
        return image

    def _pad(self, image: Image.Image, new_width: int, new_height: int) -> Image.Image:
        mean_per_channel = tuple(
            np.clip(np.array(image).mean(axis=(0, 1)), 0, 255).astype(np.uint8)
        )
        new_im = Image.new(mode="RGB", size=(new_height, new_width), color=(0, 0, 0))  # type: ignore
        new_im.paste(image)
        return new_im

    def _split(self, image: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        # Split image into number of required tiles (width x height)
        num_channels, height, width = image.size()
        image = image.view(num_channels, nch, height // nch, ncw, width // ncw)
        # Permute dimensions to reorder the axes
        image = image.permute(1, 3, 0, 2, 4).contiguous()
        # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
        image = image.view(ncw * nch, num_channels, height // nch, width // ncw)
        return image

    def _fit_image_to_canvas(
        self, num_chunks: int, img_width: int, img_height: int
    ) -> Any:
        """
        Given an image width, height and target number of chunks this function will see if the image
        can be fit into any of the canvases that can be build from arranging the tiles in a grid.
        If the image can be fit onto several canvases, it will return the canvas where the shorter edge
        of the image will be largest.
        """
        # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
        optimal_canvas = None
        optimal_image_width_height = None

        scale = img_width / img_height

        # Gather all potential supported image resolutions and iterate through them to find best match
        potential_arrangements = [
            item
            for sublist in self._find_supported_aspect_ratios(num_chunks).values()
            for item in sublist
        ]
        current_gap = 1e23
        for n_w, n_h in potential_arrangements:
            # Compute the canvas size
            canvas_width, canvas_height = n_w * self.size, n_h * self.size

            # Check if image can fit into the canvas without downsampling
            if canvas_width >= img_width and canvas_height >= img_height:
                # If we did not find a good canvas yet, we will use the current one
                if optimal_canvas is None:
                    # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                    optimal_canvas = (n_w, n_h)
                    optimal_image_width_height = (n_w * self.size, n_h * self.size)
                else:
                    # Find closest fit based on gap
                    image_width_height = (n_w * self.size, n_h * self.size)
                    gap = abs(img_width - image_width_height[0]) + abs(
                        img_height - image_width_height[1]
                    )
                    if gap < current_gap:
                        # If the gap is smaller than the previous one, we will update our optimal canvas and image width height
                        optimal_canvas = (n_w, n_h)
                        optimal_image_width_height = image_width_height
                        current_gap = gap
        return optimal_canvas

    def __call__(self, image: Image.Image, max_num_chunks: int) -> Tuple[Any, Any]:
        assert max_num_chunks > 0
        assert isinstance(image, Image.Image), type(image)
        w, h = image.size
        # Check if the image can be fit to the canvas without downsampling
        ar = self._fit_image_to_canvas(
            num_chunks=max_num_chunks, img_width=w, img_height=h
        )
        if ar is None:
            # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
            ar = self._find_closest_aspect_ratio(
                num_chunks=max_num_chunks, img_width=w, img_height=h
            )
            image = self._resize(image, ar[0] * self.size, ar[1] * self.size)
        else:
            image = self._resize_max_side_to_size(image)
        image = self._pad(image, ar[1] * self.size, ar[0] * self.size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self._split(image, ar[0], ar[1])  # type: ignore
        return image, ar

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llama_image_tokens)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llamavl)
class LlamaVLForCausalLM(nn.Module, SupportsMultiModal):
    def __init__(self, config,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        print("config", type(config))
        print(config)
        print("multimodal_config", type(multimodal_config))
        print(multimodal_config)
        print("cache_config", type(cache_config))
        print(cache_config)
        print("quant_config", type(quant_config))
        print(quant_config)

        # self.params = args
        args = config
        self.model_dim = args.dim
        self.vision_model = CrossAttentionTransformerVision(args)
        self.text_model = CrossAttentionTransformerText(args, cache_config=cache_config)
        self.image_res = args.vision_chunk_size
        self.max_num_chunks = args.vision_max_num_chunks
        self.image_transform = partial(
            VariableSizeImageTransform(size=args.vision_chunk_size),
            max_num_chunks=args.vision_max_num_chunks,
        )
        self.lm_head = ParallelLMHead(
            args.vocab_size,
            args.dim,
            org_num_embeddings=args.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            quant_config=quant_config,
        )
        self.logits_processor = LogitsProcessor(args.dim, args.vocab_size)
        self.sampler = Sampler()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        state_dict = {name: weight for name, weight in weights}
        state_dict.pop('text_model.rope.freqs')
        state_dict['lm_head.weight'] = state_dict.pop('text_model.output.weight')
        self.load_state_dict(state_dict, strict=True)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        aspect_ratios = kwargs.pop("aspect_ratios", None)

        if pixel_values is None and image_embeds is None:
            return None
        
        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided.")

        if pixel_values is not None:
            # tensor with the same shape will be batched together by MultiModalInputs.batch, so pixel_values here can be: 
            #   - List[List[torch.Tensor]]: with shape (num_chunks, 3, image_res, image_res)
            #   - List[torch.Tensor]: with shape (num_image_in_batch, num_chunks, 3, image_res, image_res)
            #   - torch.Tensor: with shape (bs, num_image_in_batch, num_chunks, 3, image_res, image_res)
            # the best choice is to remove MultiModalInputs.batch
            pixel_values_unpacked = []
            for b in range(len(pixel_values)):
                pixel_values_unpacked_b = []
                for i in range(len(pixel_values[b])):
                    pixel_values_unpacked_b.append(pixel_values[b][i])
                pixel_values_unpacked.append(pixel_values_unpacked_b)
            
            max_num_images = max([len(x) for x in pixel_values_unpacked])
            max_num_chunks = max(max([len(x) for x in y]) for y in pixel_values_unpacked)
            bsz = len(pixel_values_unpacked)
            out_num_chunks = []
            out_images = torch.zeros(
                bsz,
                max_num_images,
                max_num_chunks,
                3,
                self.image_res,
                self.image_res
            )
            out_ar = torch.ones(bsz, max_num_images, 2, dtype=torch.int64)
            for b in range(len(pixel_values_unpacked)):
                _num_chunks = []
                for i in range(len(pixel_values_unpacked[b])):
                    img = pixel_values_unpacked[b][i]
                    out_images[b, i, :img.shape[0]] = img
                    out_ar[b, i] = aspect_ratios[b][i]
                    _num_chunks.append(img.shape[0])
                out_num_chunks.append(_num_chunks)

            return LlamaImagePixelInputs(
                type="pixel_values",
                data=out_images,
                num_chunks=out_num_chunks,
                aspect_ratios=out_ar,
            )

        if image_embeds is not None:
            raise NotImplementedError

        raise AssertionError("This line should be unreachable.")

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        image_inputs = self._parse_and_validate_image_input(**kwargs)
        if image_inputs is None:
            cross_attention_masks = None
            full_text_row_masked_out_mask = None
            xattn_caches = None
        else:
            # llama's reference implementation runs the vision model on CPU
            cuda_images = image_inputs['data'].cuda()
            cuda_aspect_ratios = image_inputs['aspect_ratios'].cuda()
            vision_tokens = self.vision_model(cuda_images, cuda_aspect_ratios)
            batch_masks = []
            # TODO: get the sequence of each query without hack? 1) better attn metadata 2) better input processor to create vision mask during preprocess
            # assert isinstance(attn_metadata, PagedAttentionMetadata)
            start_pos = 0
            for seq_len in attn_metadata.seq_lens_tensor:
                end_pos = start_pos + seq_len
                batch_masks.append(create_vision_mask(input_ids[start_pos:end_pos]))
                start_pos = end_pos

            bsz, nimg, nchunk, ntok, image_token_dim = tuple(vision_tokens.shape)
            xattn_caches = torch.stack(
                [
                    layer.compute_xattn_kv_cache(
                        vision_tokens.view(bsz, -1, image_token_dim)
                    )
                    for layer in self.text_model.cross_attention_layers
                ]
            )
            # TODO: remove this hardcode
            total_len = 512
            padded_masks = _pad_masks(
                batch_masks,
                image_inputs['num_chunks'],
                total_len,
                self.max_num_chunks,
            )

            cross_attention_masks, full_text_row_masked_out_mask = (
                self.text_model._get_xattn_mask(
                    num_tokens=total_len,
                    text_device="cuda",
                    text_dtype=next(self.text_model.parameters()).dtype,
                    vision_tokens=vision_tokens,
                    cross_attention_masks=padded_masks,
                )
            )

            full_text_row_masked_out_mask_plain = torch.zeros(attn_metadata.num_prefill_tokens, 1, dtype=full_text_row_masked_out_mask.dtype)
            start_pos = 0
            for i, seq_len in enumerate(attn_metadata.seq_lens_tensor):
                end_pos = start_pos + seq_len
                full_text_row_masked_out_mask_plain[start_pos:end_pos, 0] = full_text_row_masked_out_mask[i, 0, :seq_len, 0]
                start_pos = end_pos
            full_text_row_masked_out_mask = full_text_row_masked_out_mask_plain.cuda()
        h = self.text_model.get_partially_trainable_embedding(input_ids)
        logits = self.text_model.forward(
            positions=positions,
            h=h,
            xattn_mask=cross_attention_masks,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            xattn_caches=xattn_caches,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        # import pdb; pdb.set_trace()
        return logits

def create_vision_mask(
    tokens: List[int],
    vision_token: int=128256,
) -> List[List[int]]:
    # import pdb; pdb.set_trace()
#     (Pdb) p tokens
# [128011, 128011, 128000, 644, 264, 11914, 11, 1521, 1403, 5448, 6308]
    vision_token_locations = [
        i for i, token in enumerate(tokens) if token == vision_token
    ]
    if len(vision_token_locations) == 0:
        return []

    if len(vision_token_locations) == 1:
        # only one image present, unmask until end of sequence
        return [[vision_token_locations[0], -1]]
    vision_masks = [
        [loc1, loc2]
        for loc1, loc2 in zip(vision_token_locations[:-1], vision_token_locations[1:])
    ]
    # last image will attend to all subsequent text
    vision_masks.append([vision_token_locations[-1], len(tokens)])

    # if there are two or more consecutive vision tokens,
    # they should all attend to all subsequent
    # text present
    last_mask_end = vision_masks[-1][1]
    for vision_mask in vision_masks[::-1]:
        if vision_mask[0] == vision_mask[1] - 1:
            vision_mask[1] = last_mask_end
        last_mask_end = vision_mask[1]
    return vision_masks



def _pad_masks(
    all_masks: List[List[List[int]]],
    all_num_chunks: List[List[int]],
    total_len: int,
    max_num_chunks: int,
) -> torch.Tensor:
    dtype = torch.bfloat16
    inf_value = torch.finfo(dtype).min

    bsz = len(all_masks)
    max_num_media = max([len(m) for m in all_masks])

    out_masks = torch.full(
        (bsz, total_len, max_num_media, max_num_chunks),
        inf_value,
        dtype=dtype,
    )

    for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
        for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
            if len(mask_elem) == 2:
                mask_elem[1] = min(mask_elem[1], total_len)
                if mask_elem[1] == -1:
                    mask_elem[1] = total_len
                out_masks[
                    idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks
                ].fill_(0.0)

    return out_masks
