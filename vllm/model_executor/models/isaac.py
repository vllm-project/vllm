# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Annotated, Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import TensorType
from typing_extensions import TypedDict, Unpack

from vllm.config import MultiModalConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.siglip import SiglipMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.transformers_utils.config import patch_rope_parameters
from vllm.transformers_utils.configs import (
    IsaacConfig,
    PixelShuffleSiglip2VisionConfig,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape


def create_cumulative_seq_lengths(
    seq_sizes: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create cumulative sequence lengths for variable-length attention."""
    cu_seqlens = torch.zeros(len(seq_sizes) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = seq_sizes.cumsum(0)
    max_seqlen = (
        seq_sizes.max()
        if len(seq_sizes) > 0
        else torch.tensor(0, dtype=torch.int32, device=device)
    )
    return cu_seqlens, max_seqlen


class Siglip2VariableSequenceEmbeddings(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = ReplicatedLinear(
            input_size=config.num_channels * self.patch_size * self.patch_size,
            output_size=self.embed_dim,
            return_bias=False,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def positional_embeddings(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size, -1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        _seq_patches, _seq_sizes, spatial_shapes = packed_seq_patches
        pos_embeds_list = []
        mode = "bilinear"
        align_corners = False
        antialias = True
        for spatial_shape in spatial_shapes:
            height, width = int(spatial_shape[0]), int(spatial_shape[1])
            # Guard to ensure height and width are positive for torch.compile
            if height > 0 and width > 0:
                resized_pos_embed = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode=mode,
                    align_corners=align_corners,
                    antialias=antialias,
                )
                # Reshape from (1, embed_dim, height, width) to
                # (height*width, embed_dim)
                resized_pos_embed = resized_pos_embed.reshape(
                    self.embed_dim, height * width
                ).transpose(0, 1)
            else:
                # Fallback - should never happen in practice
                resized_pos_embed = positional_embeddings.reshape(
                    self.embed_dim,
                    self.position_embedding_size * self.position_embedding_size,
                ).transpose(0, 1)[: height * width]
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings along the sequence dimension
        pos_embeds = torch.cat(pos_embeds_list, dim=0)
        return pos_embeds

    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        seq_patches, _seq_sizes, _spatial_shapes = packed_seq_patches

        target_weight = self.patch_embedding.weight
        seq_patches = seq_patches.to(
            device=target_weight.device, dtype=target_weight.dtype
        )
        patch_embeds = self.patch_embedding(seq_patches)
        pos_embeds = self.positional_embeddings(packed_seq_patches)

        # Flatten patch embeddings to match positional embeddings format
        if patch_embeds.dim() == 3:
            patch_embeds = patch_embeds.view(-1, patch_embeds.size(-1))

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + pos_embeds
        return embeddings


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
    if not torch.compiler.is_compiling() and not (
        (token_grids[:, 0] % r == 0).all() and (token_grids[:, 1] % r == 0).all()
    ):
        raise AssertionError(
            "Every (H,W) in `token_grids` must be divisible by "
            f"scale_factor={r}, got {token_grids.tolist()}"
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
            Concatenated vision embeddings. Accepts `(seq_len, hidden_size)` or
            `(1, seq_len, hidden_size)` shapes produced by stacking image
            patches.
        token_grids (`torch.Tensor`):
            Integer tensor of shape `(num_images, 2)` whose rows give the
            `(height, width)` patch grid sizes corresponding to each image
            segment inside `x`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor specific to pixel shuffle. Values
            greater than one merge `scale_factor**2` neighboring patches into a
            single embedding channel-group.

    Returns:
        `torch.Tensor`: Pixel-shuffled embeddings with shape matching the input
        convention: `(seq_len, hidden_size * scale_factor**2)` when the input
        was 2D, or `(1, seq_len, hidden_size * scale_factor**2)` if the
        singleton batch dimension was present.

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
        raise ValueError(
            f"Image (w={image.width}, h={image.height}) > MAX=`{MAX_PIXELS}`"
        )
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
            Upper bound on `(height / patch_size) * (width / patch_size)` after
            resizing.
        min_num_patches (`int`, *optional*):
            Lower bound on the number of patches. When provided the image will
            be scaled up if necessary.
        eps (`float`, *optional*, defaults to 1e-5):
            Convergence tolerance for the internal binary search to determine
            the target dimensions.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Additional stride multiplier applied when pixel shuffle later
            reduces spatial resolution.

    Returns:
        `tuple[int, int]`: Height and width (in pixels) that are multiples of
        `patch_size * pixel_shuffle_scale` and respect both the maximum and
        optional minimum patch-count constraints.
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
            target_height = get_scaled_image_size(
                scale, image_height, patch_size, pixel_shuffle_scale
            )
            target_width = get_scaled_image_size(
                scale, image_width, patch_size, pixel_shuffle_scale
            )
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(
            scale, image_height, patch_size, pixel_shuffle_scale
        )
        target_width = get_scaled_image_size(
            scale, image_width, patch_size, pixel_shuffle_scale
        )
        return target_height, target_width
    elif num_patches <= max_num_patches:
        return adjusted_height, adjusted_width
    else:
        # Scale down
        scale_min, scale_max = eps / 10, 1.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(
                scale, image_height, patch_size, pixel_shuffle_scale
            )
            target_width = get_scaled_image_size(
                scale, image_width, patch_size, pixel_shuffle_scale
            )
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale
        scale = scale_min
        target_height = get_scaled_image_size(
            scale, image_height, patch_size, pixel_shuffle_scale
        )
        target_width = get_scaled_image_size(
            scale, image_width, patch_size, pixel_shuffle_scale
        )
        return target_height, target_width


_MEAN_TENSOR = torch.tensor(VISION_MEAN, dtype=torch.float32).view(1, 1, 1, -1)
_STD_TENSOR = torch.tensor(VISION_STD, dtype=torch.float32).view(1, 1, 1, -1)


def _resolve_vision_token_id(model_config: ModelConfig, vision_token: str) -> int:
    tokenizer_name = model_config.tokenizer or model_config.model
    tokenizer = get_cached_tokenizer(
        get_tokenizer(
            tokenizer_name,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.tokenizer_revision or model_config.revision,
        )
    )
    return tokenizer.encode(vision_token, add_special_tokens=False)[0]


def prepare_image_tensor(
    image: torch.Tensor,
    scale: float = VISION_SCALE,
) -> torch.Tensor:
    r"""Standardize RGB images prior to patch extraction via rescaling and whitening.

    Args:
        image (`torch.Tensor`):
            Tensor with shape `(..., height, width, 3)` containing RGB values.
            The tensor is converted to floating point if needed.
        scale (`float`, *optional*, defaults to `VISION_SCALE`):
            Scalar multiplier applied before normalization.
    Returns:
        `torch.Tensor`: Normalized tensor with the same shape as the input and
        dtype `torch.float32`.
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
            Patch tensor where each position stores the flattened pixels
            belonging to that patch.

    Raises:
        ValueError: If `height` or `width` is not divisible by `patch_size`.
    """
    num_images, height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            "Dimensions of images "
            f"{image.shape} are not divisible by patch_size={patch_size}."
        )
    patches = image.reshape(
        num_images,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(
        num_images,
        height // patch_size,
        width // patch_size,
        channels * patch_size * patch_size,
    )
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
            Either `(height, width, channels)` for a single image or
            `(num_images, height, width, channels)` for a batch. Channels are
            expected to be RGB.
        patch_size (`int`):
            Edge length of square patches; implictly controls resize grid granularity.
        max_num_patches (`int`):
            Maximum number of patches allowed after resizing.
        min_num_patches (`int`, *optional*):
            Minimum number of patches. If provided, the routine upsamples images
            as needed to satisfy the lower bound.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Pixel shuffle scale factor; influences the target grid that the
            function produces.

    Returns:
        `tuple[torch.Tensor, list[int]]`: A pair `(patches, dims_virtual)`
        where `patches` has shape `(num_images, target_h / patch_size, target_w
        / patch_size, channels * patch_size**2)` and `dims_virtual` encodes
        effective `(images, height, width)` dimensions after optional pixel
        shuffling.
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


class IsaacImageProcessorKwargs(TypedDict, total=False):
    patch_size: int
    max_num_patches: int
    min_num_patches: int
    pixel_shuffle_scale: int


class IsaacImageProcessor:
    patch_size = 16
    max_num_patches = 6144
    min_num_patches = 256
    pixel_shuffle_scale = 2

    valid_kwargs = IsaacImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, kwargs):
        self.patch_size = kwargs.pop("patch_size", self.patch_size)
        self.vision_max_num_patches = kwargs.pop(
            "vision_max_num_patches", self.max_num_patches
        )
        self.vision_min_num_patches = kwargs.pop(
            "vision_min_num_patches", self.min_num_patches
        )
        self.pixel_shuffle_scale = kwargs.pop("pixel_shuffle_scale", 2)

    def preprocess(
        self,
        images: list[torch.Tensor],
        return_tensors: str | TensorType | None,
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> BatchFeature:
        """Preprocess images into format compatibile with vLLM input processing."""

        all_pixel_values: list[torch.Tensor] = []
        all_image_grids: list[torch.Tensor] = []

        for image in images:
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
            image_grid_thw = torch.tensor(dims_real).unsqueeze(0)

            all_pixel_values.append(pixel_values)
            all_image_grids.append(image_grid_thw)

        if all_pixel_values:
            final_pixel_values = torch.cat(all_pixel_values, dim=0)
            final_image_grids = torch.cat(all_image_grids, dim=0)
        else:
            final_pixel_values = torch.empty(0, 0)
            final_image_grids = torch.empty(0, 3)

        return BatchFeature(
            data={
                "pixel_values": final_pixel_values,
                "image_grid_thw": final_image_grids,
            },
            tensor_type=return_tensors,
        )


class IsaacProcessor:
    """Processor wrapper (tokenizer + IsaacImageProcessor)."""

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        self.image_token = kwargs.pop("image_token", "<image>")
        self.image_processor = image_processor or IsaacImageProcessor(kwargs)
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, **kwargs) -> BatchFeature:
        result = {}

        if images is not None:
            image_inputs = self.image_processor.preprocess(images, **kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            result.update(image_inputs)

            if text is not None:
                if not isinstance(text, list):
                    text = [text]

                text = text.copy()  # below lines change text in-place
                merge_length = self.image_processor.pixel_shuffle_scale**2
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        num_image_tokens = image_grid_thw[index].prod() // merge_length
                        text[i] = text[i].replace(
                            self.image_token, "<|placeholder|>" * num_image_tokens, 1
                        )
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if text is not None:
            result.update(self.tokenizer(text, **kwargs))

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
                    "content": "".join(text_parts),
                }
                processed_messages.append(processed_message)
            else:
                # Regular text message
                processed_messages.append(message)

        return self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
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
                vision_max_num_patches=getattr(
                    original_config, "vision_max_num_patches", 256
                ),
                vision_min_num_patches=getattr(
                    original_config, "vision_min_num_patches", None
                ),
                pixel_shuffle_scale=getattr(original_config, "pixel_shuffle_scale", 1),
                max_sequence_length=getattr(
                    original_config, "max_sequence_length", 16384
                ),
                vision_token=getattr(original_config, "vision_token", "<image>"),
                vision_attn_implementation=getattr(
                    original_config, "vision_attn_implementation", None
                ),
            )
        return IsaacConfig()

    def get_hf_processor(self, **kwargs) -> IsaacProcessor:
        hf_config = self.get_hf_config()
        processor_kwargs = {
            "image_token": hf_config.vision_token,
        }
        processor_kwargs.update(kwargs)
        return self.ctx.get_hf_processor(IsaacProcessor, **processor_kwargs)

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

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        num_vision_tokens = hf_config.vision_max_num_patches // (
            hf_config.pixel_shuffle_scale**2
        )
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


class IsaacImagePixelInputs(TensorSchema):
    """
    Schema for validating Isaac image inputs.

    Dimensions:
        - np: Number of patches
        - d: Patch dimension
        - ni: Number of images

    The schema enforces:
        - pixel_values must be 2D: (num_patches, patch_dim)
        - image_grid_thw must be 2D: (num_images, 3)
          where 3 represents [T, H, W]
    """

    pixel_values: Annotated[
        torch.Tensor,
        TensorShape("np", "d"),
    ]

    image_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ni", 3),
    ]


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
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            "image_grid_thw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)

        pixel_shuffle_scale = getattr(image_processor, "pixel_shuffle_scale", 2)
        merge_length = pixel_shuffle_scale**2

        def get_replacement_isaac(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            feature_size = int(grid_thw.prod()) // merge_length
            repl_full = "<|image_pad|>" * feature_size
            return PromptUpdateDetails.select_text(repl_full, "<|image_pad|>")

        return [
            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=get_replacement_isaac,
            )
        ]


class Siglip2VisionAttention(nn.Module):
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()

        use_data_parallel = (
            multimodal_config.mm_encoder_tp_mode == "data"
            if multimodal_config
            else False
        )
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.hidden_size_per_attention_head = dist_utils.divide(
            config.hidden_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            config.num_attention_heads, self.tp_size
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_attention_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            disable_tp=use_data_parallel,
        )
        self.out_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
            multimodal_config=multimodal_config,
        )

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seq_len, bs, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=2)
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("packed variable-length attention expects batch_size=1")
        x = rearrange(hidden_states, "b s d -> s b d")
        x, _ = self.qkv_proj(x)
        q, k, v = self.split_qkv(x)
        q, k, v = (rearrange(t, "s b h d -> b s h d") for t in (q, k, v))

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.out_proj(context_layer)
        output = rearrange(output, "s b d -> b s d")
        return output


class Siglip2EncoderLayer(nn.Module):
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2VisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            multimodal_config=multimodal_config,
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        *,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    multimodal_config=multimodal_config,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: PixelShuffleSiglip2VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VariableSequenceEmbeddings(config)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            multimodal_config=multimodal_config,
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_seq_patches: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        r"""
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width)
            of the input images.
        """

        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # Get embeddings from packed sequence
        hidden_states = self.embeddings((seq_patches, seq_sizes, token_grids))

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        cu_seqlens, max_seqlen = create_cumulative_seq_lengths(
            seq_sizes, hidden_states.device
        )

        hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.post_layernorm(hidden_states)

        if self.pixel_shuffle_scale_factor > 1:
            hidden_states = pixel_shuffle_varlen(
                x=hidden_states,
                token_grids=token_grids,
                scale_factor=self.pixel_shuffle_scale_factor,
            )
        # Remove the pseudo batch dimension we added earlier
        hidden_states = hidden_states.squeeze(0)

        # return last_hidden_state
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IsaacVisionEmbedding(nn.Module):
    def __init__(
        self,
        vision_cfg: PixelShuffleSiglip2VisionConfig,
        hidden_dim: int,
        output_dim: int,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.transformer = Siglip2VisionTransformer(
            vision_cfg,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=maybe_prefix(prefix, "0"),
        )
        self.linear_fc1 = ColumnParallelLinear(
            hidden_dim,
            4 * hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "1"),
            return_bias=False,
        )
        self.act = nn.SiLU()
        self.linear_fc2 = RowParallelLinear(
            4 * hidden_dim,
            output_dim,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "3"),
            return_bias=False,
        )

    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden_states = self.transformer(packed_seq_patches)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    IsaacMultiModalProcessor,
    info=IsaacProcessingInfo,
    dummy_inputs=IsaacDummyInputsBuilder,
)
class IsaacForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, SupportsMRoPE
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.text_model.lm_head.": "language_model.lm_head.",
            "model.text_model.": "language_model.model.",
            "model.vision_embedding.0": "vision_embedding.transformer",
            "model.vision_embedding.1": "vision_embedding.linear_fc1",
            "model.vision_embedding.2": "vision_embedding.act",
            "model.vision_embedding.3": "vision_embedding.linear_fc2",
            "model.vision_embedding.": "vision_embedding.",
            "model.lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config: IsaacConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        head_dim = config.head_dim
        calculated_mrope_section = [
            head_dim // 4,  # 2x more for temporal dim
            head_dim // 8,
            head_dim // 8,
        ]

        self.vision_token_id = _resolve_vision_token_id(
            vllm_config.model_config, config.vision_token
        )
        config.image_token_id = self.vision_token_id

        text_cfg = getattr(config, "text_config", None)
        target_cfg = (
            text_cfg
            if text_cfg is not None and not isinstance(text_cfg, dict)
            else config
        )

        rope_scaling = getattr(target_cfg, "rope_scaling", None)
        if rope_scaling is None and target_cfg is config:
            rope_scaling = getattr(config, "_rope_scaling", None)

        patch_rope_parameters(target_cfg)
        rope_parameters = target_cfg.rope_parameters
        rope_parameters["mrope_section"] = calculated_mrope_section
        if rope_scaling is not None and "mrope_interleaved" in rope_scaling:
            rope_parameters.setdefault(
                "mrope_interleaved", rope_scaling["mrope_interleaved"]
            )
        target_cfg.rope_parameters = rope_parameters
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            architectures=["Qwen3ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        vision_cfg = config.vision_config
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")
        attn_impl = (
            config.vision_attn_implementation
            if config.vision_attn_implementation is not None
            else getattr(config, "_attn_implementation", None)
        )
        if attn_impl is not None:
            vision_cfg._attn_implementation = attn_impl

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)
        self.vision_embedding = IsaacVisionEmbedding(
            vision_cfg=vision_cfg,
            hidden_dim=hidden_dim,
            output_dim=config.hidden_size,
            quant_config=quant_config,
            multimodal_config=self.multimodal_config,
            prefix=maybe_prefix(prefix, "vision_embedding"),
        )

    def iter_mm_grid_hw(
        self, input_tokens: list[int], mm_features: list[MultiModalFeatureSpec]
    ) -> Iterator[tuple[int, int, int]]:
        spatial_merge_size = self.config.vision_config.pixel_shuffle_scale_factor
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                yield offset, h // spatial_merge_size, w // spatial_merge_size
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        llm_pos_ids_list = []
        st = 0
        for offset, llm_grid_h, llm_grid_w in self.iter_mm_grid_hw(
            input_tokens, mm_features
        ):
            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
            grid_indices[0, :] = grid_indices[0, :] + text_len + st_idx
            llm_pos_ids_list.append(grid_indices)
            st = offset + llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1][0, -1] + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

        return torch.from_numpy(llm_positions), mrope_position_delta

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> IsaacImagePixelInputs | None:
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            return None

        # TensorSchema will automatically validate shapes on initialization
        return IsaacImagePixelInputs(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def _process_image_input(
        self,
        image_input: IsaacImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values"]
        image_grid_thw = image_input["image_grid_thw"]
        if pixel_values.numel() == 0:
            return ()

        device = next(self.language_model.parameters()).device
        dtype = self.vision_embedding.linear_fc1.weight.dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        spatial_grids = image_grid_thw[:, 1:3].to(device, dtype=torch.int32)

        vision_embeddings = self.vision_embedding((pixel_values, spatial_grids))
        merge_size = self.config.vision_config.pixel_shuffle_scale_factor
        sizes = spatial_grids.prod(-1) // (merge_size * merge_size)
        return tuple(vision_embeddings.split(sizes.tolist()))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return ()
        return self._process_image_input(image_input)

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> MultiModalEmbeddings | None:
        # Backward compatibility for older runners.
        embeddings = self.embed_multimodal(**kwargs)
        if not embeddings:
            return []
        return embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="vision_embedding.linear_fc2",  # The final linear layer
            tower_model="vision_embedding",
        )
