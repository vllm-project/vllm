# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard tests for DeepSeek-OCR _encode_local_features.

_encode_local_features must skip processing only for an empty-element
patches tensor, not for a zero-sum one.

Regression for #64075: the old guard `torch.sum(patches).item() == 0`
returned None for a 1536x768 half-white/half-black image whose two
normalised crop patches cancel to sum==0, causing a shape mismatch in
_merge_multimodal_embeddings and a fatal EngineCore crash.

The correct guard is `patches.numel() == 0`.
"""

import pytest
import torch
from transformers import CLIPVisionConfig

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    finalize_model_weights,
)
from vllm.model_executor.models.deepencoder import (
    DeepCLIPVisionEmbeddings,
    ImageEncoderViT,
)
from vllm.model_executor.models.deepseek_ocr import DeepseekOCRForCausalLM
from vllm.model_executor.models.deepseek_ocr2 import DeepseekOCR2ForCausalLM
from vllm.model_executor.models.step3_vl import Step3VisionEmbeddings
from vllm.model_executor.models.vision import interpolate_cls_abs_pos
from vllm.transformers_utils.configs.step3_vl import Step3VisionEncoderConfig


class TestDeepseekOCREncodeLocalFeaturesGuard:
    def test_ocr2_empty_patches_returns_none(self):
        """Zero-element tensor → skip local encoding (non-tiled image)."""
        empty = torch.empty(0, 3, 224, 224)
        assert DeepseekOCR2ForCausalLM._encode_local_features(None, empty) is None

    def test_ocr2_zero_sum_patches_enters_model_body(self):
        """Non-empty patches that happen to sum to zero must NOT return None.

        The method will raise AttributeError when it tries to call
        self.sam_model on a None self – that is the expected outcome here
        because it proves the guard was passed, not that encoding succeeded.
        The real crash was a shape-mismatch ValueError that killed EngineCore.
        """
        # Two patches that cancel: white (1.0) + black (-1.0) after Normalize.
        wb = torch.cat([torch.ones(1, 3, 4, 4), -torch.ones(1, 3, 4, 4)])
        assert wb.numel() > 0
        assert wb.sum().item() == 0.0
        with pytest.raises(AttributeError):
            DeepseekOCR2ForCausalLM._encode_local_features(None, wb)

    def test_ocr1_empty_patches_returns_none(self):
        """OCR-1: zero-element tensor → skip local encoding."""
        empty = torch.empty(0, 3, 224, 224)
        crop_shape = torch.tensor([0, 0])
        assert (
            DeepseekOCRForCausalLM._encode_local_features(None, empty, crop_shape)
            is None
        )

    def test_ocr1_zero_sum_patches_enters_model_body(self):
        """OCR-1: non-empty zero-sum patches must pass the guard."""
        wb = torch.cat([torch.ones(1, 3, 4, 4), -torch.ones(1, 3, 4, 4)])
        crop_shape = torch.tensor([1, 2])
        with pytest.raises(AttributeError):
            DeepseekOCRForCausalLM._encode_local_features(None, wb, crop_shape)


def _make_small_sam() -> ImageEncoderViT:
    sam = ImageEncoderViT(
        img_size=64,
        patch_size=8,
        embed_dim=16,
        depth=1,
        num_heads=2,
        out_chans=8,
        use_abs_pos=True,
    ).eval()
    assert sam.pos_embed is not None
    with torch.no_grad():
        sam.pos_embed.copy_(torch.randn_like(sam.pos_embed))
    return sam


@pytest.mark.core_model
@pytest.mark.usefixtures("default_vllm_config")
class TestAbsPosEmbedCache:
    def test_sam_cache_hit_and_bit_exact(self):
        sam = _make_small_sam()
        assert sam.pos_embed is not None
        expected = ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, 6)

        with torch.inference_mode():
            out1 = sam.get_abs_pos(sam.pos_embed, 6)
            out2 = sam.get_abs_pos(sam.pos_embed, 6)

        assert torch.equal(out1, expected)
        assert out1 is out2
        assert out1.is_contiguous()

        # Non-member tensor bypasses instance cache.
        other = torch.randn_like(sam.pos_embed)
        other_out = sam.get_abs_pos(other, 6)
        assert other_out is not out1
        assert torch.equal(other_out, ImageEncoderViT._interpolate_abs_pos(other, 6))

    def test_sam_refresh_in_place_after_default_weight_loader(self):
        """`default_weight_loader` writes via `.data.copy_` (no `_version` bump),
        so `finalize_model_weights` must refresh cached entries in place while
        preserving `data_ptr()` for captured encoder CUDA/NPU graphs."""
        sam = _make_small_sam()
        assert sam.pos_embed is not None

        with torch.inference_mode():
            before = sam.get_abs_pos(sam.pos_embed, 6)
        cached_ptr = before.data_ptr()

        new_weight = torch.randn_like(sam.pos_embed) + 5.0
        default_weight_loader(sam.pos_embed, new_weight)
        finalize_model_weights(sam)

        expected = ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, 6)
        with torch.inference_mode():
            after = sam.get_abs_pos(sam.pos_embed, 6)

        assert torch.equal(after, expected)
        assert after.data_ptr() == cached_ptr

    def test_sam_invalidates_on_copy_and_param_replacement(self):
        sam = _make_small_sam()
        assert sam.pos_embed is not None

        with torch.inference_mode():
            first = sam.get_abs_pos(sam.pos_embed, 6)
        first_ptr = first.data_ptr()

        with torch.no_grad():
            sam.pos_embed.copy_(torch.randn_like(sam.pos_embed) + 3.0)
        with torch.inference_mode():
            second = sam.get_abs_pos(sam.pos_embed, 6)
        assert torch.equal(
            second, ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, 6)
        )
        assert second.data_ptr() == first_ptr

        sam.pos_embed = torch.nn.Parameter(
            torch.randn_like(sam.pos_embed) - 3.0, requires_grad=False
        )
        with torch.inference_mode():
            third = sam.get_abs_pos(sam.pos_embed, 6)
        assert torch.equal(
            third, ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, 6)
        )

    def test_sam_cache_bounded_without_evicting_live_entries(self):
        sam = _make_small_sam()
        assert sam.pos_embed is not None

        with torch.inference_mode():
            pinned = sam.get_abs_pos(sam.pos_embed, 4)
            for size in range(5, 18):
                if size == sam.pos_embed.size(1):
                    continue
                out = sam.get_abs_pos(sam.pos_embed, size)
                assert torch.equal(
                    out, ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, size)
                )
            again = sam.get_abs_pos(sam.pos_embed, 4)

        assert len(sam._abs_pos_cache._entries) == 8
        assert again is pinned

    def test_deepclip_and_step3_embeddings_cache_and_reload(self):
        clip_cfg = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            image_size=32,
            patch_size=8,
        )
        clip_emb = DeepCLIPVisionEmbeddings(clip_cfg).eval()
        step3_cfg = Step3VisionEncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=1,
            image_size=32,
            patch_size=8,
        )
        step3_emb = Step3VisionEmbeddings(step3_cfg).eval()

        for module, tgt_arg, tgt_grid in [
            (clip_emb, 3 * 3 + 1, 3),
            (step3_emb, 3 * 3, 3),
        ]:
            expected = interpolate_cls_abs_pos(
                module.position_embedding(module.position_ids), tgt_grid
            )
            with torch.inference_mode():
                out1 = module._get_pos_embed(tgt_arg)
                out2 = module._get_pos_embed(tgt_arg)
            assert torch.equal(out1, expected)
            assert out1 is out2
            cached_ptr = out1.data_ptr()

            new_w = torch.randn_like(module.position_embedding.weight) + 2.0
            default_weight_loader(module.position_embedding.weight, new_w)
            finalize_model_weights(module)

            expected_new = interpolate_cls_abs_pos(
                module.position_embedding(module.position_ids), tgt_grid
            )
            with torch.inference_mode():
                out3 = module._get_pos_embed(tgt_arg)
            assert torch.equal(out3, expected_new)
            assert out3.data_ptr() == cached_ptr

            # Verify full forward matches uncached interpolation.
            pixel_values = torch.randn(2, 3, 24, 24)
            with torch.inference_mode():
                y1 = module(pixel_values)
                y2 = module(pixel_values)
            assert torch.equal(y1, y2)

    def test_torch_compile_bypasses_cache(self):
        sam = _make_small_sam()
        assert sam.pos_embed is not None

        compiled = torch.compile(sam.get_abs_pos, backend="eager")
        with torch.inference_mode():
            out = compiled(sam.pos_embed, 6)

        assert len(sam._abs_pos_cache._entries) == 0
        assert torch.equal(out, ImageEncoderViT._interpolate_abs_pos(sam.pos_embed, 6))
