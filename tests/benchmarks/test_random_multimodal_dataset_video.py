# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import os
import tempfile
from typing import Any, Dict, List, Optional, cast

import cv2
import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import RandomMultiModalDataset, SampleRequest


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    """Use a small, commonly available tokenizer."""
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def video_dataset() -> RandomMultiModalDataset:
    """Create a RandomMultiModalDataset instance for testing."""
    return RandomMultiModalDataset(random_seed=42)


class TestVideoSampling:
    """Test video sampling functionality in RandomMultiModalDataset."""

    @pytest.mark.benchmark
    def test_generate_synthetic_video_basic(self, video_dataset: RandomMultiModalDataset):
        """Test basic synthetic video generation."""
        width, height, num_frames = 64, 48, 8
        
        result = video_dataset.generate_synthetic_video(width, height, num_frames)
        
        # Check return type and structure
        assert isinstance(result, dict)
        assert "bytes" in result
        assert isinstance(result["bytes"], bytes)
        assert len(result["bytes"]) > 0
        
        # Verify the video can be decoded
        video_bytes = result["bytes"]
        
        # Create a temporary file to test video decoding
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_bytes)
        
        try:
            # Test that OpenCV can read the video
            cap = cv2.VideoCapture(temp_path)
            assert cap.isOpened(), "Generated video should be readable by OpenCV"
            
            # Check video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            assert frame_count == num_frames, f"Expected {num_frames} frames, got {frame_count}"
            assert frame_width == width, f"Expected width {width}, got {frame_width}"
            assert frame_height == height, f"Expected height {height}, got {frame_height}"
            
            cap.release()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.benchmark
    def test_generate_synthetic_video_different_sizes(self, video_dataset: RandomMultiModalDataset):
        """Test video generation with different dimensions and frame counts."""
        test_cases = [
            (32, 32, 4),   # Small square video
            (128, 64, 16), # Rectangular video
            (64, 128, 32), # Tall video
            (256, 256, 8), # Large square video
        ]
        
        for width, height, num_frames in test_cases:
            result = video_dataset.generate_synthetic_video(width, height, num_frames)
            
            assert isinstance(result, dict)
            assert "bytes" in result
            assert len(result["bytes"]) > 0
            
            # Verify video properties
            video_bytes = result["bytes"]
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_bytes)
            
            try:
                cap = cv2.VideoCapture(temp_path)
                assert cap.isOpened()
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                assert frame_count == num_frames
                assert frame_width == width
                assert frame_height == height
                
                cap.release()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    @pytest.mark.benchmark
    def test_generate_synthetic_video_deterministic(self, video_dataset: RandomMultiModalDataset):
        """Test that video generation is deterministic with same seed."""
        width, height, num_frames = 64, 48, 8
        
        # Generate two videos with same parameters
        video1 = video_dataset.generate_synthetic_video(width, height, num_frames)
        video2 = video_dataset.generate_synthetic_video(width, height, num_frames)
        
        # Videos should be identical due to deterministic random seed
        assert video1["bytes"] == video2["bytes"]

    @pytest.mark.benchmark
    def test_generate_synthetic_video_different_seeds(self):
        """Test that different seeds produce different videos."""
        dataset1 = RandomMultiModalDataset(random_seed=123)
        dataset2 = RandomMultiModalDataset(random_seed=456)
        
        width, height, num_frames = 64, 48, 8
        
        video1 = dataset1.generate_synthetic_video(width, height, num_frames)
        video2 = dataset2.generate_synthetic_video(width, height, num_frames)
        
        # Videos should be different due to different seeds
        assert video1["bytes"] != video2["bytes"]

    @pytest.mark.benchmark
    def test_map_config_to_modality(self, video_dataset: RandomMultiModalDataset):
        """Test modality mapping for different configurations."""
        # Test image configuration (num_frames = 1)
        assert video_dataset.map_config_to_modality((256, 256, 1)) == "image"
        assert video_dataset.map_config_to_modality((720, 1280, 1)) == "image"
        
        # Test video configurations (num_frames > 1)
        assert video_dataset.map_config_to_modality((256, 256, 8)) == "video"
        assert video_dataset.map_config_to_modality((720, 1280, 16)) == "video"
        assert video_dataset.map_config_to_modality((64, 64, 32)) == "video"
        
        # Test invalid configurations
        with pytest.raises(ValueError, match="Invalid multimodal item configuration"):
            video_dataset.map_config_to_modality((256, 256, 0))
        
        with pytest.raises(ValueError, match="Invalid multimodal item configuration"):
            video_dataset.map_config_to_modality((256, 256, -1))

    @pytest.mark.benchmark
    def test_generate_mm_item_video(self, video_dataset: RandomMultiModalDataset):
        """Test generating multimodal items for video configurations."""
        # Test video item generation
        video_config = (64, 48, 8)  # height, width, num_frames
        result = video_dataset.generate_mm_item(video_config)
        
        # Check the result structure matches OpenAI API format
        assert isinstance(result, dict)
        assert result["type"] == "video_url"
        assert "video_url" in result
        assert "url" in result["video_url"]
        
        # Check that the URL is a data URL with base64 encoded video
        url = result["video_url"]["url"]
        assert url.startswith("data:video/mp4;base64,")
        
        # Decode and verify the video content
        base64_data = url.split(",")[1]
        video_bytes = base64.b64decode(base64_data)
        assert len(video_bytes) > 0
        
        # Verify the video can be decoded
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_bytes)
        
        try:
            cap = cv2.VideoCapture(temp_path)
            assert cap.isOpened()
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            assert frame_count == 8
            assert frame_width == 48
            assert frame_height == 64
            
            cap.release()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.benchmark
    def test_generate_mm_item_image(self, video_dataset: RandomMultiModalDataset):
        """Test generating multimodal items for image configurations."""
        # Test image item generation
        image_config = (64, 48, 1)  # height, width, num_frames=1
        result = video_dataset.generate_mm_item(image_config)
        
        # Check the result structure matches OpenAI API format
        assert isinstance(result, dict)
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert "url" in result["image_url"]
        
        # Check that the URL is a data URL with base64 encoded image
        url = result["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    @pytest.mark.benchmark
    def test_generate_mm_item_invalid_config(self, video_dataset: RandomMultiModalDataset):
        """Test error handling for invalid configurations."""
        with pytest.raises(ValueError, match="Invalid multimodal item configuration"):
            video_dataset.generate_mm_item((256, 256, 0))

    @pytest.mark.benchmark
    def test_sample_with_video_buckets(self, video_dataset: RandomMultiModalDataset, 
                                     hf_tokenizer: PreTrainedTokenizerBase):
        """Test sampling with video bucket configurations."""
        # Configure bucket with video probability > 0
        bucket_config = {
            (64, 64, 1): 0.3,   # Images
            (64, 64, 8): 0.7,   # Videos
        }
        
        limit_mm_per_prompt = {"image": 5, "video": 3}
        
        samples = video_dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=5,
            base_items_per_request=2,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        assert len(samples) == 5
        
        # Check that samples contain both images and videos
        video_count = 0
        image_count = 0
        
        for sample in samples:
            assert isinstance(sample, SampleRequest)
            assert sample.multi_modal_data is not None
            assert isinstance(sample.multi_modal_data, list)
            
            mm_data = cast(List[Dict[str, Any]], sample.multi_modal_data)
            assert len(mm_data) == 2  # base_items_per_request
            
            for item in mm_data:
                if item["type"] == "video_url":
                    video_count += 1
                    # Verify video URL format
                    url = item["video_url"]["url"]
                    assert url.startswith("data:video/mp4;base64,")
                elif item["type"] == "image_url":
                    image_count += 1
                    # Verify image URL format
                    url = item["image_url"]["url"]
                    assert url.startswith("data:image/jpeg;base64,")
        
        # Should have some videos due to 0.7 probability
        assert video_count > 0
        assert image_count > 0

    @pytest.mark.benchmark
    def test_sample_video_only_buckets(self, video_dataset: RandomMultiModalDataset,
                                      hf_tokenizer: PreTrainedTokenizerBase):
        """Test sampling with only video buckets."""
        bucket_config = {
            (64, 64, 8): 1.0,   # Only videos
        }
        
        limit_mm_per_prompt = {"image": 0, "video": 2}
        
        samples = video_dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=3,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        assert len(samples) == 3
        
        for sample in samples:
            assert isinstance(sample, SampleRequest)
            assert sample.multi_modal_data is not None
            assert isinstance(sample.multi_modal_data, list)
            
            mm_data = cast(List[Dict[str, Any]], sample.multi_modal_data)
            assert len(mm_data) == 1
            
            item = mm_data[0]
            assert item["type"] == "video_url"
            url = item["video_url"]["url"]
            assert url.startswith("data:video/mp4;base64,")

    @pytest.mark.benchmark
    def test_sample_respects_video_limits(self, video_dataset: RandomMultiModalDataset,
                                        hf_tokenizer: PreTrainedTokenizerBase):
        """Test that sampling respects video limits per prompt."""
        bucket_config = {
            (64, 64, 8): 1.0,   # Only videos
        }
        
        # Set very low video limit
        limit_mm_per_prompt = {"image": 0, "video": 1}
        
        samples = video_dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=3,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        assert len(samples) == 3
        
        for sample in samples:
            mm_data = cast(List[Dict[str, Any]], sample.multi_modal_data)
            assert len(mm_data) <= 1  # Should respect video limit

    @pytest.mark.benchmark
    def test_sample_mixed_buckets_with_zero_probability(self, video_dataset: RandomMultiModalDataset,
                                                      hf_tokenizer: PreTrainedTokenizerBase):
        """Test sampling with mixed buckets including zero probability entries."""
        bucket_config = {
            (64, 64, 1): 0.5,   # Images
            (64, 64, 8): 0.5,   # Videos
            (128, 128, 16): 0.0, # Zero probability videos (should be ignored)
        }
        
        limit_mm_per_prompt = {"image": 2, "video": 2}
        
        samples = video_dataset.sample(
            tokenizer=hf_tokenizer,
            num_requests=4,
            base_items_per_request=2,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        assert len(samples) == 4
        
        # Should only see 64x64 videos, not 128x128 videos
        for sample in samples:
            mm_data = cast(List[Dict[str, Any]], sample.multi_modal_data)
            for item in mm_data:
                if item["type"] == "video_url":
                    # Decode video to verify dimensions
                    url = item["video_url"]["url"]
                    base64_data = url.split(",")[1]
                    video_bytes = base64.b64decode(base64_data)
                    
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(video_bytes)
                    
                    try:
                        cap = cv2.VideoCapture(temp_path)
                        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                        
                        # Should be 64x64, not 128x128
                        assert frame_width == 64
                        assert frame_height == 64
                    finally:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

    @pytest.mark.benchmark
    def test_sample_deterministic_with_videos(self, hf_tokenizer: PreTrainedTokenizerBase):
        """Test that sampling with videos is deterministic with same seed."""
        dataset1 = RandomMultiModalDataset(random_seed=123)
        dataset2 = RandomMultiModalDataset(random_seed=123)
        
        bucket_config = {
            (64, 64, 1): 0.3,   # Images
            (64, 64, 8): 0.7,   # Videos
        }
        
        limit_mm_per_prompt = {"image": 2, "video": 2}
        
        samples1 = dataset1.sample(
            tokenizer=hf_tokenizer,
            num_requests=3,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        samples2 = dataset2.sample(
            tokenizer=hf_tokenizer,
            num_requests=3,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        assert len(samples1) == len(samples2)
        
        # Compare multimodal data
        for s1, s2 in zip(samples1, samples2):
            assert s1.multi_modal_data == s2.multi_modal_data

    @pytest.mark.benchmark
    def test_sample_different_seeds_produce_different_videos(self, hf_tokenizer: PreTrainedTokenizerBase):
        """Test that different seeds produce different video content."""
        dataset1 = RandomMultiModalDataset(random_seed=123)
        dataset2 = RandomMultiModalDataset(random_seed=456)
        
        bucket_config = {
            (64, 64, 8): 1.0,   # Only videos
        }
        
        limit_mm_per_prompt = {"image": 0, "video": 1}
        
        samples1 = dataset1.sample(
            tokenizer=hf_tokenizer,
            num_requests=2,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        samples2 = dataset2.sample(
            tokenizer=hf_tokenizer,
            num_requests=2,
            base_items_per_request=1,
            num_mm_items_range_ratio=0.0,
            limit_mm_per_prompt=limit_mm_per_prompt,
            bucket_config=bucket_config,
            input_len=20,
            output_len=5,
        )
        
        # Video content should be different
        for s1, s2 in zip(samples1, samples2):
            mm_data1 = cast(List[Dict[str, Any]], s1.multi_modal_data)
            mm_data2 = cast(List[Dict[str, Any]], s2.multi_modal_data)
            
            assert len(mm_data1) == len(mm_data2) == 1
            
            url1 = mm_data1[0]["video_url"]["url"]
            url2 = mm_data2[0]["video_url"]["url"]
            
            assert url1 != url2  # Different video content
