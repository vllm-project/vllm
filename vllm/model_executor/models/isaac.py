from __future__ import annotations

from collections.abc import Mapping, Sequence, Iterable
from typing import Any, Optional, Union
from typing_extensions import TypedDict, Unpack

import itertools
from enum import Enum
from dataclasses import dataclass

import math
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, Qwen3Config
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import TensorType
from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2MLP,
)
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (
    WeightsMapper,
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.parse import MultiModalDataItems, ImageSize
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargs,
    MultiModalDataDict,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)

# ===== TensorStream Compatibility Layer for Isaac MRoPE =====
# Minimal implementation of TensorStream classes needed for Isaac's 3D positional encoding

class ModalityType(Enum):
    """
    Base class for modality-type enumerations.
    Each derived class (VisionType, TextType) holds
    an integer value that identifies a specific modality.

    Example usage:
        If you have an object `my_event` of class `Event`,
        you might write:
            if my_event.type == VisionType.image:
                # process an image frame

    The methods below implement ordering and hashing
    based on the integer `.value` of each enum member.
    """

    @property
    def modality(self):
        return self.__class__

    def __lt__(self, other):
        if isinstance(other, ModalityType):
            return self.value < other.value
        raise NotImplementedError()

    def __eq__(self, other):
        if isinstance(other, ModalityType):
            return self.value == other.value
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.value)


# NOTE: modality types need to be unique
class VisionType(ModalityType):
    """
    Enum for vision modalities such as key video frames.
    Typically used in video processing or image sequences.

    Members:
        image: A single image frame.
    """

    image = 0


class TextType(ModalityType):
    """
    Enum for text tokens and padding.

    Members:
        text: Actual textual tokens.
        padding: Padding tokens used in sequence batching.
    """

    text = 1
    padding = 2


@dataclass
class Event:
    """Represents a single modality event with spatial/temporal dimensions."""
    """
    Represents a single data occurrence (with a specific type, time interval, and data payload).

    Attributes:
        data (Any): The actual data payload (e.g. a torch.Tensor, a string, etc.).
        type (ModalityType): The modality type of the data (e.g., VisionType.image).
        time (Tuple[float, float]): (start_time, end_time) indicating when this Event occurs.
        role (Optional[str]): The role associated with this event (e.g., "user", "agent", "system").
            If None, the event is always included in loss calculation.

    Example usage:
        evt = Event(data=torch.zeros((1, 224, 224, 3)),  # e.g. a single image frame
                    type=VisionType.image,
                    time=(0.0, 0.04),
                    role="user")
    """    
    # Descriptors
    modality_type: ModalityType
    
    # Structure
    dims_virtual: list[int] | None = None  # virtual/processed dimensions (e.g., pixel-shuffled)
    dims_real: list[int] | None = None  # real/actual tensor dimensions
    idx_range: tuple[int, int] | None = None
    
    def dims(self, virtual: bool = True) -> list[int] | None:
        """
        Get the dimensions of this event.

        Args:
            virtual: If True (default), return virtual/processed dimensions (e.g., pixel-shuffled).
                    If False, return real/actual tensor dimensions.

        Returns:
            Dimensions list or None if not measured.
        """
        if virtual:
            return self.dims_virtual
        else:
            return self.dims_real

    def num_tokens(self, partial=True, virtual=True) -> int:
        if not virtual:
            assert partial is False and isinstance(self.data, torch.Tensor)
            return math.prod(self.dims(virtual=False))
        return self.idx_range[1] - self.idx_range[0] if partial else math.prod(self.dims())


@dataclass
class Stream:
    """
    Represents an ordered sequence of Event objects, each with
    a specific ModalityType and a time range.

    Attributes:
        events (List[Event]): The list of Event objects in the stream.
        priority (List[ModalityType]): A list of modality types that define
            how we might want to reorder or prioritize events if scheduling is needed.

    Example usage:
        # Create two events of different types
        evt1 = Event(torch.zeros((1, 224, 224, 3)), VisionType.image, (0.0, 0.04))
        evt2 = Event(torch.randint(0, 1000, (16, 1)), TextType.text, (0.0, 0.32))

        # Make a stream with a given priority
        s = Stream(events=[evt1, evt2],
                   priority=[VisionType.image, TextType.text])

        print(s)
    """

    events: list[Event]

    def __len__(self):
        """Returns the number of Event objects in this Stream."""
        return len(self.events)

    def __getitem__(self, key: int) -> Stream | Event:
        return self.events[key]

    def __iter__(self):
        """
        Yields each Event in the Stream, enabling iteration like:
            for event in my_stream:
                ...
        """
        yield from self.events


# TODO: implement all types of cool indexing which can happen since TensorStream assuems Event.data = Tensor
@dataclass
class TensorStream:
    streams: list[Stream]
    _device: torch.device | None = None

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        seq_lens = [sum([ev.num_tokens() for ev in stream]) for stream in self.streams]
        assert all([sl == seq_lens[0] for sl in seq_lens]), (
            f"each stream must have same token count to have a shape: {seq_lens}"
        )
        return (len(seq_lens), seq_lens[0])


def compute_mrope_pos_tensor(ts: TensorStream, n_pos_dims: int = 3) -> torch.Tensor:
    """
    Create a (batch, T, n_pos_dims) position tensor in one sweep.
    The first dim is the running “time” index, the rest are spatial (or 1-fillers).

    Args:
        ts         : TensorStream
        n_pos_dims : total coordinate dimensions (default 3)

    Returns:
        torch.LongTensor  - shape (batch_size, seq_len, n_pos_dims)
    """

    # Manually iterate through streams and events like map_compact does,
    # but maintain cumulative time offset for each stream
    all_coords = []
    for stream in ts.streams:  # one Stream == one batch sample
        cumulative_offset = 0  # running time index for this stream

        for event in stream:
            # --- build coordinate grid for THIS event using itertools (no tensor ops) ---
            dims = (event.dims() or [1]) + [1] * (n_pos_dims - len(event.dims() or []))

            # Create ranges for each dimension (similar to old _finalize implementation)
            first_dim = range(cumulative_offset, cumulative_offset + dims[0])
            cumulative_offset += dims[0]  # advance time for the next event
            other_dims = [range(d) for d in dims[1:]]

            # Use itertools.product to create all coordinate combinations
            full_coords = list(itertools.product(first_dim, *other_dims))

            # Slice if the event is partial
            s, e = event.idx_range
            coords = full_coords[s:e]

            # Extend the flattened coordinate list
            all_coords.extend(coords)

    # Convert to tensor and reshape to (B, T, n_pos_dims)
    B, T = ts.shape
    return torch.tensor(all_coords, dtype=torch.long, device=ts.device).reshape(B, T, n_pos_dims)


def modality_mask(ts: TensorStream, modality_type: ModalityType) -> torch.Tensor:
    """Create boolean mask for specific modality type in the tensor stream."""
    B, T = ts.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=ts.device)
    
    for batch_idx, stream in enumerate(ts.streams):
        seq_idx = 0
        for event in stream:
            if event.modality_type == modality_type:
                start, end = event.idx_range
                mask[batch_idx, seq_idx:seq_idx+(end-start)] = True
            seq_idx += (event.idx_range[1] - event.idx_range[0])
    
    return mask

# ===== End TensorStream Compatibility Layer =====

class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.
    """

    model_type = "pixel_shuffle_siglip2"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor
        self.num_patches = num_patches


def create_cumulative_seq_lengths(seq_sizes: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    """Create cumulative sequence lengths for variable-length attention."""
    cu_seqlens = torch.zeros(len(seq_sizes) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = seq_sizes.cumsum(0)
    max_seqlen = int(seq_sizes.max().item()) if len(seq_sizes) > 0 else 0
    return cu_seqlens, max_seqlen


class Siglip2VariableSequenceEmbeddings(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def positional_embeddings(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        _seq_patches, _seq_sizes, spatial_shapes = packed_seq_patches
        pos_embeds_list = []
        mode = "bilinear"
        align_corners = False
        antialias = True
        for spatial_shape in spatial_shapes:
            height, width = spatial_shape
            # Guard to ensure height and width are positive for torch.compile
            if height > 0 and width > 0:
                resized_pos_embed = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode=mode,
                    align_corners=align_corners,
                    antialias=antialias,
                )
                # Reshape from (1, embed_dim, height, width) to (height*width, embed_dim)
                resized_pos_embed = resized_pos_embed.reshape(self.embed_dim, height * width).transpose(0, 1)
            else:
                # Fallback - should never happen in practice
                resized_pos_embed = positional_embeddings.reshape(
                    self.embed_dim, self.position_embedding_size * self.position_embedding_size
                ).transpose(0, 1)[: height * width]
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings along the sequence dimension
        pos_embeds = torch.cat(pos_embeds_list, dim=0)
        return pos_embeds

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        seq_patches, _seq_sizes, _spatial_shapes = packed_seq_patches

        # Apply patch embeddings
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(seq_patches.to(dtype=target_dtype))
        pos_embeds = self.positional_embeddings(packed_seq_patches)

        # Flatten patch embeddings to match positional embeddings format
        # From [batch, patches_per_image, embed_dim] to [total_patches, embed_dim]
        batch_size, patches_per_image, embed_dim = patch_embeds.shape

        # For variable-length attention, we need to reshape to (total_tokens, embed_dim)
        if batch_size != 1:
            raise ValueError("Variable-length attention expects batch_size=1 for packed sequences")

        patch_embeds = patch_embeds.view(batch_size * patches_per_image, embed_dim)

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + pos_embeds
        return embeddings


class Siglip2VariableLengthAttention(nn.Module):
    """Custom attention that supports variable-length sequences with flash attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
        batch_size, seq_len, _ = hidden_states.size()

        # For variable-length attention, we need to reshape to (total_tokens, embed_dim)
        if batch_size != 1:
            raise ValueError("Variable-length attention expects batch_size=1 for packed sequences")
        hidden_states = hidden_states.squeeze(0)  # Remove batch dimension: (seq_len, embed_dim)

        # Store original dtype
        orig_dtype = hidden_states.dtype

        # 1. Linear projections
        Q = self.q_proj(hidden_states)  # (seq_len, embed_dim)
        K = self.k_proj(hidden_states)  # (seq_len, embed_dim)
        V = self.v_proj(hidden_states)  # (seq_len, embed_dim)

        # 2. Reshape for multi-head attention: (seq_len, n_heads, head_dim)
        Q = Q.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(-1, self.num_heads, self.embed_dim // self.num_heads)

        # 3. Apply variable-length attention using flash attention
        attn_output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            query=Q,
            key=K,
            value=V,
            cum_seq_q=cu_seqlens,
            cum_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=self.scale,
            window_size_left=-1,
            window_size_right=-1,
            alibi_slopes=None,
        )

        # 4. Reshape attention output from (seq_len, n_heads, head_dim) to (seq_len, embed_dim)
        attn_output = attn_output.reshape(seq_len, self.embed_dim)

        # 5. Convert back to original dtype if needed
        if attn_output.dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype)

        # 6. Project output
        attn_output = self.out_proj(attn_output)  # (seq_len, embed_dim)

        # 7. Add back batch dimension for compatibility
        attn_output = attn_output.unsqueeze(0)  # (1, seq_len, embed_dim)

        return attn_output, None


class IsaacSiglip2EncoderLayer(nn.Module):
    """Siglip2 encoder layer with variable-length attention."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2VariableLengthAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)  # Use HF's Siglip2MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class IsaacEncoder(nn.Module):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([IsaacSiglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, None


def create_pixel_shuffle_index_map(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a gather-index map that tells us, for every *output* token after
    pixel-shuffle, which `scale_factor**2` *input* tokens are being merged.

    Args
    ----
    seq_sizes     : (num_images,)  - #patches in each image (row-major order)
    token_grids   : (num_images,2) - (height, width) for every image
    scale_factor  : spatial down-scale factor (≥2)
    device        : (optional) overrides `seq_sizes.device`

    Returns
    -------
    gather_idx : (new_total_seq_len, scale_factor**2) int64 tensor.
                 gather_idx[i, j] is the *flat* index into the *original*
                 packed sequence for the j-th sub-patch that forms the
                 i-th output token.
    """
    if device is None:
        device = seq_sizes.device

    r = int(scale_factor)
    if r < 2:
        raise ValueError("`scale_factor` must be ≥ 2")

    # Safety: all spatial dims must be divisible by r
    # Cannot run under torch compile fullgraph mode hence
    if not torch.compiler.is_compiling():
        if not ((token_grids[:, 0] % r == 0).all() and (token_grids[:, 1] % r == 0).all()):
            raise AssertionError(
                f"Every (H,W) in `token_grids` must be divisible by scale_factor={r}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        # Build the (H, W) grid of flat indices for this image
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)  # (H, W)

        # -------- identical ordering to your fixed-res routine --------
        # Step 1: split width into blocks of r
        grid = grid.view(h, w // r, r)  # (H, W/r, r)
        # Step 2: now split height into blocks of r
        grid = grid.view(h // r, r, w // r, r)  # (H/r, r, W/r, r)
        # Step 3: final permutation to (H/r, W/r, r, r)
        grid = grid.permute(0, 2, 1, 3).contiguous()  # (H/r, W/r, r, r)
        # Step 4: each (r, r) block forms one output token
        gather_chunks.append(grid.reshape(-1, r * r))  # (H*W / r², r²)

        tok_offset += seq_len

    # Concatenate over all images in the packed batch
    gather_idx = torch.cat(gather_chunks, dim=0)  # (Σ_i HᵢWᵢ/r², r²)
    return gather_idx


def pixel_shuffle_varlen(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
) -> torch.Tensor:
    r"""Apply pixel shuffle to a packed vision sequence without unpacking per image.

    Args:
        x (`torch.Tensor`):
            Concatenated vision embeddings. Accepts `(seq_len, hidden_size)` or `(1, seq_len, hidden_size)` shapes
            produced by stacking image patches.
        token_grids (`torch.Tensor`):
            Integer tensor of shape `(num_images, 2)` whose rows give the `(height, width)` patch grid sizes
            corresponding to each image segment inside `x`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor specific to pixel shuffle. Values greater than one merge `scale_factor**2` neighboring patches into a
            single embedding channel-group.

    Returns:
        `torch.Tensor`: Pixel-shuffled embeddings with shape matching the input convention:
        `(seq_len, hidden_size * scale_factor**2)` when the input was 2D, or `(1, seq_len, hidden_size * scale_factor**2)`
        if the singleton batch dimension was present.

    Raises:
        ValueError: If more than one batch item is provided.
    """
    keep_batch_dim = x.dim() == 3
    if keep_batch_dim:
        if x.size(0) != 1:
            raise AssertionError("Packed sequence is expected to have batch_size == 1")
        x_ = x.squeeze(0)  # (seq, embed)
    else:
        x_ = x  # (seq, embed)

    embed_dim = x_.size(-1)
    r = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build index map and gather in one go
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=r,
        device=x_.device,
    )  # (new_seq, r²)

    # Gather → (new_seq, r², embed_dim)
    gathered = x_[gather_idx]  # fancy indexing keeps gradient

    # Merge the r² group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * r * r)

    # Restore batch dimension if needed
    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


class Siglip2SequenceVisionTransformer(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VariableSequenceEmbeddings(config)
        self.encoder = IsaacEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # Get embeddings from packed sequence
        hidden_states = self.embeddings((seq_patches, seq_sizes, token_grids))

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens, max_seqlen = create_cumulative_seq_lengths(seq_sizes, hidden_states.device)

        # Pass through encoder with variable-length attention parameters
        hidden_states, _, _ = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Apply final layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        if self.pixel_shuffle_scale_factor > 1:
            hidden_states = pixel_shuffle_varlen(
                x=hidden_states,
                token_grids=token_grids,
                scale_factor=self.pixel_shuffle_scale_factor,
            )
        # Remove the pseudo batch dimension we added earlier
        hidden_states = hidden_states.squeeze(0)

        # Return the full sequence of embeddings
        return hidden_states


# ============================================================================
# Configuration
# ============================================================================

MAX_PIXELS = 60_000_000  # 60-megapixel ceiling ≈ 8200 × 7300 px

# Vision preprocessing constants
VISION_MEAN = (0.5, 0.5, 0.5)
VISION_STD = (0.5, 0.5, 0.5)
VISION_SCALE = 1 / 255


def _make_writeable(arr: np.ndarray) -> np.ndarray:
    """Return *arr* itself if it is already writeable, otherwise try to flip the
    write flag in-place and finally fall back to `arr.copy()`.
    This guarantees the buffer handed to `torch.from_numpy()` is always
    writeable, silencing the PyTorch warning about undefined behaviour.
    """
    if arr.flags.writeable:
        return arr

    # First, try the cheap path — in-place flag toggle (works for mmap'd arrays
    # and some shared memory buffers):
    try:
        arr.setflags(write=True)
        return arr  # success: no data copy
    except ValueError:
        # Buffer is inherently read-only (e.g. backed by PyAV / PIL): make copy
        return arr.copy()


def extract_image_pil(image: PIL.Image.Image) -> torch.Tensor | None:
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"Image (w={image.width}, h={image.height}) > MAX=`{MAX_PIXELS}`")
    img = image if image.mode == "RGB" else image.convert("RGB")
    arr = np.asarray(img)
    arr = _make_writeable(arr)
    return torch.from_numpy(arr)


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]:
    r"""Compute a target resolution whose patch grid satisfies patching parametrization.

    Args:
        image_height (`int`):
            Height in pixels of the source image prior to any resizing.
        image_width (`int`):
            Width in pixels of the source image prior to any resizing.
        patch_size (`int`):
            Size of the square patch used by the vision encoder.
        max_num_patches (`int`):
            Upper bound on `(height / patch_size) * (width / patch_size)` after resizing.
        min_num_patches (`int`, *optional*):
            Lower bound on the number of patches. When provided the image will be scaled up if necessary.
        eps (`float`, *optional*, defaults to 1e-5):
            Convergence tolerance for the internal binary search to determing the target dimensions.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Additional stride multiplier applied when pixel shuffle later reduces spatial resolution.

    Returns:
        `tuple[int, int]`: Height and width (in pixels) that are multiples of `patch_size * pixel_shuffle_scale`
        and respect both the maximum and optional minimum patch-count constraints.
    """

    def get_scaled_image_size(scale, original_size, patch_size, pixel_shuffle_scale):
        scaled_size = scale * original_size
        divisor = patch_size * pixel_shuffle_scale
        scaled_size = math.ceil(scaled_size / divisor) * divisor
        scaled_size = max(divisor, scaled_size)
        return int(scaled_size)

    # Ensure divisibility
    divisor = patch_size * pixel_shuffle_scale
    adjusted_height = math.ceil(image_height / divisor) * divisor
    adjusted_height = max(divisor, adjusted_height)
    adjusted_width = math.ceil(image_width / divisor) * divisor
    adjusted_width = max(divisor, adjusted_width)

    num_patches = (adjusted_height / patch_size) * (adjusted_width / patch_size)

    if min_num_patches is not None and num_patches < min_num_patches:
        # Scale up
        scale_min, scale_max = 1.0, 100.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width
    elif num_patches <= max_num_patches:
        return adjusted_height, adjusted_width
    else:
        # Scale down
        scale_min, scale_max = eps / 10, 1.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale
        scale = scale_min
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width


_MEAN_TENSOR = torch.tensor(VISION_MEAN, dtype=torch.float32).view(1, 1, 1, -1)
_STD_TENSOR = torch.tensor(VISION_STD, dtype=torch.float32).view(1, 1, 1, -1)


def prepare_image_tensor(
    image: torch.Tensor,
    scale: float = VISION_SCALE,
) -> torch.Tensor:
    r"""Standardize RGB images prior to patch extraction via rescaling and whitening.

    Args:
        image (`torch.Tensor`):
            Tensor with shape `(..., height, width, 3)` containing RGB values. The tensor is converted to floating
            point if needed.
        scale (`float`, *optional*, defaults to `VISION_SCALE`):
            Scalar multiplier applied before normalization.
    Returns:
        `torch.Tensor`: Normalized tensor with the same shape as the input and dtype `torch.float32`.
    """
    if not torch.is_floating_point(image):
        image = image.float()
    rescaled = image * scale

    # Use precomputed tensors and move to the correct device if needed
    mean_tensor = _MEAN_TENSOR.to(image.device)
    std_tensor = _STD_TENSOR.to(image.device)

    normalized = (rescaled - mean_tensor) / std_tensor
    return normalized


def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""Convert normalized images into flattened ViT-style patches.

    Args:
        image (`torch.Tensor`):
            Tensor of shape `(num_images, height, width, channels)`.
        patch_size (`int`):
            Edge length of the square patches

    Returns:
        `torch.Tensor`:
            Patch tensor where each position stores the flattened pixels belonging to that patch.

    Raises:
        ValueError: If `height` or `width` is not divisible by `patch_size`.
    """
    num_images, height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Dimensions of images {image.shape} are not divisible by patch_size={patch_size}.")
    patches = image.reshape(num_images, height // patch_size, patch_size, width // patch_size, patch_size, channels)
    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(num_images, height // patch_size, width // patch_size, channels * patch_size * patch_size)
    return patches


def process_vision_for_patches(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    pixel_shuffle_scale: int = 1,
) -> tuple[torch.Tensor, list[int]]:
    r"""Resize, normalize, and patchify RGB images for the vision encoder.

    Args:
        images (`torch.Tensor`):
            Either `(height, width, channels)` for a single image or `(num_images, height, width, channels)` for a
            batch. Channels are expected to be RGB.
        patch_size (`int`):
            Edge length of square patches; implictly controls resize grid granularity.
        max_num_patches (`int`):
            Maximum number of patches allowed after resizing.
        min_num_patches (`int`, *optional*):
            Minimum number of patches. If provided, the routine upsamples images as needed to satisfy the lower bound.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            pixel shuffle scale factor; influences the target grid that the function produces.

    Returns:
        `tuple[torch.Tensor, list[int]]`: A pair `(patches, dims_virtual)` where `patches` has shape
        `(num_images, target_h / patch_size, target_w / patch_size, channels * patch_size**2)` and `dims_virtual`
        encodes effective `(images, height, width)` dimensions after optional pixel shuffling.
    """
    # Add batch dim if single image
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Permute to channel first for resize
    images = images.permute(0, 3, 1, 2)

    # Get target dimensions
    _, _, orig_height, orig_width = images.shape
    target_height, target_width = get_image_size_for_max_num_patches(
        orig_height,
        orig_width,
        patch_size,
        max_num_patches,
        min_num_patches=min_num_patches,
        pixel_shuffle_scale=pixel_shuffle_scale,
    )

    # Resize
    images = F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Back to channel last
    images = images.permute(0, 2, 3, 1)

    # Normalize
    images = prepare_image_tensor(images)

    # Patchify
    patches = patchify_vision(images, patch_size=patch_size)

    # Calculate dimensions for the patches
    n_images, h_patches, w_patches, _ = patches.shape
    dims_virtual = (
        [1, h_patches, w_patches]
        if pixel_shuffle_scale == 1
        else [1, h_patches // pixel_shuffle_scale, w_patches // pixel_shuffle_scale]
    )

    return patches, dims_virtual


class IsaacConfig(Qwen3Config):
    """Configuration class for Isaac multimodal model."""

    model_type = "isaac"
    sub_configs = {"vision_config": PixelShuffleSiglip2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        max_sequence_length: int = 16384,
        vision_token: str = "<|image_pad|>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # EventStreamProcessor parameters (for backward compatibility)
        self.video_patch_size = vision_patch_size
        self.vision_max_num_patches = vision_max_num_patches
        self.vision_min_num_patches = vision_min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token

        # Handle vision config - PixelShuffleSiglip2VisionConfig instance
        self.vision_config = PixelShuffleSiglip2VisionConfig(
            pixel_shuffle_scale_factor=pixel_shuffle_scale,
            num_patches=vision_max_num_patches,
        )


class IsaacImageProcessorKwargs(TypedDict, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int
    #merge_size: int  # kept for parity with other processors that expose it


class IsaacImageProcessor:

    patch_size = 16
    max_num_patches = 6144
    min_num_patches = 256
    pixel_shuffle_scale = 2

    valid_kwargs = IsaacImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, kwargs):
        self.patch_size = kwargs.pop("patch_size", self.patch_size)
        self.vision_max_num_patches = kwargs.pop("vision_max_num_patches", self.max_num_patches)
        self.vision_min_num_patches = kwargs.pop("vision_min_num_patches", self.min_num_patches)
        self.pixel_shuffle_scale = kwargs.pop("pixel_shuffle_scale", 2)

    def preprocess(
        self,
        images: list[torch.Tensor],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> BatchFeature:
        """Isaac's resize → normalize → patchify → pack."""

        all_pixel_values: list[torch.Tensor] = []
        all_image_grids: list[torch.Tensor] = []

        for image  in images:
            image_tensor = extract_image_pil(image)
            
            patches, dims_virtual = process_vision_for_patches(
                image_tensor,
                patch_size=self.patch_size,
                max_num_patches=self.vision_max_num_patches,
                min_num_patches=self.vision_min_num_patches,
                pixel_shuffle_scale=self.pixel_shuffle_scale,
            )

            # Isaac packs a dummy temporal dim for images
            patches = patches.unsqueeze(1)  # [N, T=1, Hp, Wp, D]

            hp, wp, dim = patches.shape[-3], patches.shape[-2], patches.shape[-1]
            current_num_patches = hp * wp
            pixel_values = patches.reshape(current_num_patches, dim)  # [N_tokens, D]

            # Use real patch dimensions for image_grid_thw, not virtual dimensions
            # This ensures the vision model receives correct grid info for pixel shuffle
            dims_real = [1, hp, wp]  # Real patch dimensions
            image_grid_thw = torch.tensor(dims_real).unsqueeze(0)  # [1, [T, H, W]]

            all_pixel_values.append(pixel_values)
            all_image_grids.append(image_grid_thw)

        if all_pixel_values:
            final_pixel_values = torch.cat(all_pixel_values, dim=0)
            final_image_grids = torch.cat(all_image_grids, dim=0)
        else:
            final_pixel_values = torch.empty(0, 0)
            final_image_grids = torch.empty(0, 3)

        return BatchFeature(
            data={"pixel_values": final_pixel_values, "image_grid_thw": final_image_grids},
            tensor_type=return_tensors,
        )


class IsaacProcessor:
    """Processor wrapper (tokenizer + IsaacImageProcessor)."""

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_processor = image_processor or IsaacImageProcessor(kwargs)
        self.tokenizer = tokenizer
        self.image_token = "<|image_pad|>"

    def __call__(self, text=None, images=None, **kwargs) -> BatchFeature:
        result = {}

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))
        if images is not None:
            image_result = self.image_processor.preprocess(images, **kwargs)
            result.update(image_result)
        return BatchFeature(result)
    
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        # Convert mixed content messages to simple text format
        processed_messages = []
        
        for message in messages:
            if "content" in message and isinstance(message["content"], list):
                # Handle mixed content (text + image)
                text_parts = []
                for content_item in message["content"]:
                    if content_item.get("type") == "text":
                        text_parts.append(content_item.get("text", ""))
                    elif content_item.get("type") == "image":
                        # Replace image with vision token
                        text_parts.append(self.image_token)
                
                processed_message = {
                    "role": message.get("role", "user"),
                    "content": "".join(text_parts)
                }
                processed_messages.append(processed_message)
            else:
                # Regular text message
                processed_messages.append(message)
        
        return self.tokenizer.apply_chat_template(
            processed_messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )


class IsaacProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> IsaacConfig:
        if hasattr(self.ctx, "get_hf_config"):
            original_config = self.ctx.get_hf_config()
            # Map HF config parameters to our vLLM config parameters
            return IsaacConfig(
                # Vision parameters - map from HF names
                vision_config=getattr(original_config, "vision_config", None),
                vision_patch_size=getattr(original_config, "video_patch_size", 16),
                vision_max_num_patches=getattr(original_config, "vision_max_num_patches", 256),
                vision_min_num_patches=getattr(original_config, "vision_min_num_patches", None),
                pixel_shuffle_scale=getattr(original_config, "pixel_shuffle_scale", 1),
                max_sequence_length=getattr(original_config, "max_sequence_length", 16384),
                vision_token="<|image_pad|>",
            )
        return IsaacConfig()

    def get_hf_processor(self, **kwargs) -> IsaacProcessor:
        return self.ctx.get_hf_processor(IsaacProcessor, **kwargs)

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()
        # Get target dimensions
        target_height, target_width = get_image_size_for_max_num_patches(
            9999999,
            9999999,
            hf_config.video_patch_size,
            hf_config.vision_max_num_patches,
            min_num_patches=hf_config.vision_min_num_patches,
            pixel_shuffle_scale=hf_config.pixel_shuffle_scale,
        )
        return ImageSize(width=target_width, height=target_height)

    def get_image_processor(self, **kwargs) -> IsaacImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        num_vision_tokens = hf_config.vision_max_num_patches // (hf_config.pixel_shuffle_scale**2)
        return {"image": num_vision_tokens}


class IsaacDummyInputsBuilder(BaseDummyInputsBuilder[IsaacProcessingInfo]): 
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        hf_processor = self.info.get_hf_processor()
        image_token: str = hf_processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class IsaacMultiModalProcessor(BaseMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
            
        # Configure multimodal fields for Isaac model
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)

        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:

        #hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()

        vocab = tokenizer.get_vocab()
        placeholder_id = vocab.get("<|image_pad|>", 151655)
        
        pixel_shuffle_scale = getattr(image_processor, 'pixel_shuffle_scale', 2)
        merge_length = pixel_shuffle_scale ** 2
            
        def get_replacement_isaac(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder_id] * num_tokens            

        return [
            PromptReplacement(
                modality="image",
                target=[placeholder_id],
                replacement=get_replacement_isaac,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    IsaacMultiModalProcessor,
    info=IsaacProcessingInfo,
    dummy_inputs=IsaacDummyInputsBuilder,
)
class IsaacForConditionalGeneration(
        Qwen3ForCausalLM, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_embedding.": "vision_embedding.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image_pad|>"
        
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):

        config: IsaacConfig = vllm_config.model_config.hf_config
        head_dim = config.head_dim

        calculated_mrope_section = [
            head_dim // 4,  # 2x more for temporal dim
            head_dim // 8,
            head_dim // 8,
        ]

        config.rope_scaling["mrope_section"] = calculated_mrope_section
        self.config = config

        # Initialize the parent class with updated config
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Create the language model module to match checkpoint structure
        self.language_model = nn.ModuleDict({
            "embed_tokens": self.model.embed_tokens,
            "layers": self.model.layers,
            "norm": self.model.norm
        })

        vision_cfg = config.vision_config
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)
        self.vision_embedding = nn.Sequential(
            Siglip2SequenceVisionTransformer(vision_cfg),
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, config.hidden_size, bias=False),
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: list[list[int]] | torch.Tensor,
        video_grid_thw: list[list[int]] | torch.Tensor,
        context_len: int = 0,
        seq_len: int | None = None,
        second_per_grid_ts: list[float] | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        vision_token_id = getattr(self.config, 'image_token_id', 151655)
        spatial_merge_size = hf_config.vision_config.pixel_shuffle_scale_factor
        input_tokens_tensor = torch.tensor(input_tokens)
        
        # Find image token positions
        image_positions = torch.where(input_tokens_tensor == vision_token_id)[0].tolist()
        
        # For text-only inputs, use Isaac's original logic from compute_position_ids_input_ids()
        if len(image_positions) == 0:
            seq_len = len(input_tokens)
            # Create 3D positions where all dimensions get the same 1D temporal progression
            position_ids = torch.arange(seq_len, dtype=torch.long)
            position_ids = position_ids.view(1, -1).expand(1, -1)  # [1, seq_len]
            position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # [1, seq_len, 3]

            # vLLM expects shape [3, seq_len], so transpose
            position_ids = position_ids.squeeze(0).transpose(0, 1)  # [3, seq_len]
            
            return position_ids, 0
        
        events = []
        image_idx = 0
        current_pos = 0
        last_processed_pos = -1

        for image_pos in image_positions:
            if image_pos <= last_processed_pos:
                continue  # Skip already processed positions
            
            # Add any text before this image
            if image_pos > current_pos:
                text_tokens = image_pos - current_pos
                text_event = Event(
                    modality_type=TextType.text,
                    dims_virtual=[text_tokens, 1],
                    idx_range=(0, text_tokens),
                )
                events.append(text_event)
            
            # Add image
            t, h, w = image_grid_thw[image_idx]
            llm_grid_h, llm_grid_w = h // spatial_merge_size, w // spatial_merge_size
            image_tokens = t * llm_grid_h * llm_grid_w
            
            image_event = Event(
                modality_type=VisionType.image,
                dims_virtual=[t, llm_grid_h, llm_grid_w],
                idx_range=(0, image_tokens),
            )
            events.append(image_event)
            
            current_pos = image_pos + image_tokens
            last_processed_pos = current_pos - 1  # Mark up to this position as processed
            image_idx += 1

        # Add final text segment if any
        if current_pos < len(input_tokens):
            text_tokens = len(input_tokens) - current_pos
            text_event = Event(
                modality_type=TextType.text,
                dims_virtual=[text_tokens, 1],
                idx_range=(0, text_tokens),
            )
            events.append(text_event)
        
        stream = Stream(events)
        tensor_stream = TensorStream([stream])

        # Use Isaac's native MRoPE calculation
        position_ids = compute_mrope_pos_tensor(tensor_stream, n_pos_dims=3)

        # Max position per batch across the 3 planes and sequence dimension: (B,)
        m_per_batch = position_ids.amax(dim=(1, 2))

        mrope_position_delta = (m_per_batch + 1 - len(input_tokens)).item()

        # vLLM expects shape [3, seq_len] but Isaac returns [batch, seq_len, 3]
        # Transpose to match vLLM's expected format
        position_ids = position_ids.squeeze(0).transpose(0, 1)

        return position_ids, mrope_position_delta

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings | None:    

        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")

        if pixel_values is None:
            return []

        # Convert image_grid_thw from [batch, 1, [T, H, W]] to [batch, [H, W]]
        spatial_grids = image_grid_thw[:, 0, 1:3]  # Extract H, W from [T, H, W] for each image
        
        # Process packed sequence patches through vision_embedding module
        vision_embeddings = self.vision_embedding((pixel_values, spatial_grids))

        # Split concatenated embeddings for each image item (following Qwen2-VL pattern)
        merge_size = self.config.vision_config.pixel_shuffle_scale_factor  # Isaac uses pixel shuffle
        sizes = spatial_grids.prod(-1) // (merge_size * merge_size)  # H * W / (merge_size^2)
        
        return vision_embeddings.split(sizes.tolist())

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:

        # Get text embeddings from the base language model
        inputs_embeds = super().get_input_embeddings(input_ids)
        
        # If we have multimodal embeddings, merge them with text embeddings
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        return inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.vision_embedding is None:
            skip_prefixes.extend(["vision_embedding."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="vision_embedding.3",  # The final linear layer
            tower_model="vision_embedding",
        )
