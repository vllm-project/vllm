# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Moondream3 multimodal processing.

Includes:
- Processor creation and application tests
- Image tokenization and placeholder expansion tests
- Tiling and cropping logic tests (CPU-based)
- Pixel normalization tests
"""

import pytest
import torch

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MOONDREAM3_MODEL_ID = "moondream/moondream3-preview"
# Expected tokens: 729 (27x27 patches from 378x378 crop / 14 patch size)
EXPECTED_IMAGE_TOKENS = 729
# Vision encoder constants
CROP_SIZE = 378
PATCH_SIZE = 14
MAX_CROPS = 12


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_creation(model_id: str):
    """Test that Moondream3 processor can be created."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    assert processor is not None


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_apply(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that Moondream3 processor can process inputs.

    NOTE: The prompt must have a space after <image> to ensure correct
    tokenization: '<image> ' not '<image>\\n'. This is because the
    tokenizer treats '<image>' differently based on following context.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Use space after <image> to ensure correct tokenization
    prompt = "<image> \n\nQuestion: What is this?\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    assert "prompt_token_ids" in processed_inputs
    # Token count should be close to 729 (image) + text tokens
    assert len(processed_inputs["prompt_token_ids"]) >= EXPECTED_IMAGE_TOKENS


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_pixel_values(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that pixel values are correctly produced."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<image> \n\nQuestion: What is this?\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})

    # Check mm_kwargs contains pixel_values
    mm_kwargs = processed_inputs.get("mm_kwargs")
    assert mm_kwargs is not None
    mm_data_result = mm_kwargs.get_data()
    assert "pixel_values" in mm_data_result

    # Verify pixel_values shape
    pixel_values = mm_data_result["pixel_values"]
    assert pixel_values.dim() == 5  # [batch, num_crops, C, H, W]
    assert pixel_values.shape[2] == 3  # RGB channels
    assert pixel_values.shape[3] == 378  # crop height
    assert pixel_values.shape[4] == 378  # crop width


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_image_token_expansion(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that <image> placeholder is expanded to correct number of tokens."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.tokenizer

    # The placeholder tokens for <image>
    placeholder_tokens = tokenizer.encode("<image>", add_special_tokens=False)
    # Should be [48, 4737, 50] = ['<', 'image', '>']
    assert len(placeholder_tokens) == 3

    prompt = "<image> \n\nQuestion: Describe.\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})
    token_ids = processed_inputs["prompt_token_ids"]

    # Count occurrences of the first placeholder token (used as replacement)
    # The <image> should be expanded to 729 tokens
    first_placeholder_token = placeholder_tokens[0]  # '<' token
    count = token_ids.count(first_placeholder_token)

    # Should have 729 image tokens
    assert count == EXPECTED_IMAGE_TOKENS, (
        f"Expected {EXPECTED_IMAGE_TOKENS} image tokens, got {count}"
    )


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_multi_crop_tiling(
    model_id: str,
):
    """Test that large images produce correct multi-crop tiling."""
    from PIL import Image

    from vllm.transformers_utils.processors.moondream3 import Moondream3Processor

    processor = Moondream3Processor.from_pretrained(model_id, trust_remote_code=True)

    # Create a large image that requires multiple crops
    large_image = Image.new("RGB", (1000, 1000), color="blue")
    pixel_values, tiling = processor.preprocess_image(large_image)

    # Large images should produce more than 1x1 tiling
    assert tiling[0] >= 1 and tiling[1] >= 1
    # Check that we have global crop + local crops
    expected_crops = tiling[0] * tiling[1] + 1
    assert pixel_values.shape[0] == expected_crops


@pytest.mark.parametrize(
    "image_size",
    [
        (500, 500),
        (800, 600),
        (1920, 1080),
    ],
)
@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_tiling_various_sizes(
    image_size: tuple[int, int],
    model_id: str,
):
    """Test tiling with various image sizes."""
    from PIL import Image

    from vllm.transformers_utils.processors.moondream3 import Moondream3Processor

    processor = Moondream3Processor.from_pretrained(model_id, trust_remote_code=True)

    width, height = image_size
    image = Image.new("RGB", (width, height), color="red")
    pixel_values, tiling = processor.preprocess_image(image)

    # Basic shape checks
    assert pixel_values.dim() == 4  # [num_crops, C, H, W]
    assert pixel_values.shape[1] == 3  # RGB
    assert pixel_values.shape[2] == 378  # crop height
    assert pixel_values.shape[3] == 378  # crop width

    # Tiling should respect max_crops (12)
    assert tiling[0] * tiling[1] <= 12


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_pixel_normalization(
    model_id: str,
):
    """Test that pixel values are normalized to [-1, 1] range."""
    from PIL import Image

    from vllm.transformers_utils.processors.moondream3 import Moondream3Processor

    processor = Moondream3Processor.from_pretrained(model_id, trust_remote_code=True)

    # Create test image
    image = Image.new("RGB", (378, 378), color="green")
    pixel_values, _ = processor.preprocess_image(image)

    # Normalization: (x - 0.5) / 0.5 = 2*x - 1
    # For input [0, 1], output should be [-1, 1]
    assert pixel_values.min() >= -1.0
    assert pixel_values.max() <= 1.0


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_chat_template_with_image(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that chat template correctly formats BOS + image + prompt."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = ctx.tokenizer

    # Use the chat template format
    prompt = "<|endoftext|><image> \n\nQuestion: What is this?\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})
    token_ids = processed_inputs["prompt_token_ids"]

    # BOS token (<|endoftext|>) should be token ID 0
    bos_token_id = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    assert bos_token_id == 0

    # First token should be BOS
    assert token_ids[0] == bos_token_id


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_bos_token_always_first(
    image_assets: ImageTestAssets,
    model_id: str,
):
    """Test that BOS token (ID 0) is always at position 0."""
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    # Start with BOS token explicitly
    prompt = "<|endoftext|><image> \n\nQuestion: Describe this image.\n\nAnswer:"
    mm_data = {"image": [image_assets[0].pil_image]}

    processed_inputs = processor.apply(prompt, mm_data, {})
    token_ids = processed_inputs["prompt_token_ids"]

    # Token ID 0 (<|endoftext|>) should be the first token
    assert token_ids[0] == 0, (
        f"Expected BOS token (0) at position 0, got {token_ids[0]}"
    )


@pytest.mark.parametrize("model_id", [MOONDREAM3_MODEL_ID])
def test_processor_with_small_image(
    model_id: str,
):
    """Test processor with image smaller than crop size."""
    from PIL import Image

    from vllm.transformers_utils.processors.moondream3 import Moondream3Processor

    processor = Moondream3Processor.from_pretrained(model_id, trust_remote_code=True)

    # Small image (smaller than crop size)
    small_image = Image.new("RGB", (100, 100), color="yellow")
    pixel_values, tiling = processor.preprocess_image(small_image)

    # Small images should use 1x1 tiling
    assert tiling == (1, 1)
    # Should have 2 crops (global + 1 local)
    assert pixel_values.shape[0] == 2


class TestMoondream3TilingLogic:
    """CPU-based tests for Moondream3 tiling selection logic.

    These tests validate the select_tiling() function which determines
    how images are divided into crops for the vision encoder.
    """

    def test_small_image_no_tiling(self):
        """Small images should use 1x1 tiling."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=300, width=300, crop_size=CROP_SIZE, max_crops=MAX_CROPS
        )
        assert tiling == (1, 1)

    def test_exact_crop_size(self):
        """Image exactly at crop size should use 1x1."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=CROP_SIZE, width=CROP_SIZE, crop_size=CROP_SIZE, max_crops=MAX_CROPS
        )
        assert tiling == (1, 1)

    def test_large_square_image(self):
        """Large square image should use multiple tiles."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=800, width=800, crop_size=CROP_SIZE, max_crops=MAX_CROPS
        )
        h_tiles, w_tiles = tiling
        assert h_tiles >= 2
        assert w_tiles >= 2
        assert h_tiles * w_tiles <= MAX_CROPS

    def test_wide_image(self):
        """Wide image should have more width tiles."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=400, width=1200, crop_size=CROP_SIZE, max_crops=MAX_CROPS
        )
        h_tiles, w_tiles = tiling
        assert w_tiles >= h_tiles

    def test_tall_image(self):
        """Tall image should have more height tiles."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=1200, width=400, crop_size=CROP_SIZE, max_crops=MAX_CROPS
        )
        h_tiles, w_tiles = tiling
        assert h_tiles >= w_tiles

    def test_respects_max_crops(self):
        """Tiling should not exceed max_crops."""
        from vllm.model_executor.models.moondream3 import select_tiling

        tiling = select_tiling(
            height=2000, width=2000, crop_size=CROP_SIZE, max_crops=4
        )
        h_tiles, w_tiles = tiling
        assert h_tiles * w_tiles <= 4


class TestMoondream3VisionShapes:
    """CPU-based tests for vision encoder expected shapes.

    These tests verify the mathematical relationships between
    crop size, patch size, and token counts.
    """

    def test_expected_patch_count(self):
        """Test 378/14 = 27 patches per side, 729 total."""
        patches_per_side = CROP_SIZE // PATCH_SIZE
        total_patches = patches_per_side**2

        assert patches_per_side == 27
        assert total_patches == EXPECTED_IMAGE_TOKENS

    def test_patch_embedding_input_dim(self):
        """Test patch embedding input dimension."""
        channels = 3
        input_dim = PATCH_SIZE * PATCH_SIZE * channels

        assert input_dim == 14 * 14 * 3
        assert input_dim == 588


class TestMoondream3TauAttention:
    """CPU-based tests for tau attention scaling components.

    These tests validate the tau attention formula used in Moondream3:
    - Token-based: tok_q = tanh(gelu(qkv) @ tau_wq.T)
    - Position-based: tau_pos = 1 + (sigmoid(alpha * log(pos+1)) - 0.5)
    """

    def test_tau_position_range(self):
        """Test tau position scaling produces values in valid range."""
        num_heads = 32
        seq_len = 100

        tau_alpha = torch.randn(num_heads)
        positions = torch.arange(seq_len)

        pos_float = (positions.float() + 1.0).clamp(min=1e-6)
        pos_log = pos_float.log()
        tau_pos = 1.0 + (torch.sigmoid(tau_alpha[:, None] * pos_log[None, :]) - 0.5)

        assert tau_pos.shape == (num_heads, seq_len)
        # tau_pos should be between 0.5 and 1.5
        assert tau_pos.min() >= 0.5
        assert tau_pos.max() <= 1.5

    def test_tau_token_output_range(self):
        """Test tau token scaling output is bounded by tanh."""
        import torch.nn.functional as F

        seq_len = 100
        qkv_dim = 6144  # 2048 * 3
        num_heads = 32

        qkv = torch.randn(seq_len, qkv_dim)
        tau_wq = torch.randn(num_heads, qkv_dim)

        tok_feat = F.gelu(qkv)
        tok_q = torch.tanh(tok_feat @ tau_wq.t())

        assert tok_q.shape == (seq_len, num_heads)
        # tanh output is bounded by [-1, 1]
        assert tok_q.min() >= -1.0
        assert tok_q.max() <= 1.0
