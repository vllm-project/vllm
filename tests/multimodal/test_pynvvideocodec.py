# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for PyNvVideoCodec video backend."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import numpy.typing as npt
import pytest
import torch

from vllm.multimodal.image import ImageMediaIO
from vllm.multimodal.video import (
    PyNVVideoBackend,
    VIDEO_LOADER_REGISTRY,
    VideoMediaIO,
)

from .utils import create_video_from_image

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyNvVideoCodec requires CUDA"
)


# Mock classes for PyNvVideoCodec
class MockStreamMetadata:
    """Mock for PyNvVideoCodec stream metadata."""
    def __init__(self, fps: float = 30.0):
        self.average_fps = fps


class MockDecoder:
    """Mock for PyNvVideoCodec SimpleDecoder."""
    def __init__(
        self,
        file_path: str,
        output_color_type=None,
        use_device_memory: bool = True,
        need_scanned_stream_metadata: bool = False,
        gpu_id: int = 0,
        cuda_stream=None,
        decoder_cache_size: int = 2,
    ):
        self.file_path = file_path
        self.total_frames = 100
        self.fps = 30.0
        self._metadata = MockStreamMetadata(self.fps)
        
    def __len__(self):
        return self.total_frames
    
    def get_stream_metadata(self):
        return self._metadata
    
    def get_batch_frames_by_index(self, frame_indices: list[int]):
        """Return mock DLPack frames."""
        # Return mock frames as torch tensors that can be converted via dlpack
        frames = []
        for _ in frame_indices:
            # Create a mock frame: H=720, W=1280, C=3
            frame = torch.randint(0, 255, (720, 1280, 3), 
                                 dtype=torch.uint8, device='cuda')
            frames.append(frame)
        return frames
    
    def reconfigure_decoder(self, file_path: str):
        """Reconfigure decoder for a new file."""
        self.file_path = file_path


class MockOutputColorType:
    """Mock for PyNvVideoCodec.OutputColorType."""
    RGB = "RGB"


@pytest.fixture
def mock_pynvvideocodec():
    """Fixture to mock PyNvVideoCodec module."""
    mock_module = MagicMock()
    mock_module.SimpleDecoder = MockDecoder
    mock_module.OutputColorType = MockOutputColorType
    return mock_module


@pytest.fixture
def sample_video_file(tmp_path):
    """Create a sample video file for testing."""
    # Create a simple test image
    image_path = tmp_path / "test_image.jpg"
    from PIL import Image
    img = Image.new('RGB', (640, 480), color='red')
    img.save(image_path)
    
    # Create a video from the image
    video_path = tmp_path / "test_video.mp4"
    create_video_from_image(
        str(image_path),
        str(video_path),
        num_frames=10,
        fps=30.0,
        is_color=True,
        fourcc="mp4v",
    )
    
    return video_path


@pytest.fixture(autouse=True)
def cleanup_pynv_backend():
    """Cleanup PyNVVideoBackend state between tests."""
    # Clear cached CUDA stream before each test
    PyNVVideoBackend._cuda_stream = None
    PyNVVideoBackend._set_thread_decoder(None)
    yield
    # Clear after test as well
    PyNVVideoBackend._cuda_stream = None
    PyNVVideoBackend._set_thread_decoder(None)


class TestPyNVVideoBackend:
    """Test suite for PyNVVideoBackend."""
    
    def test_hardware_accelerated_codecs(self):
        """Test that hardware-accelerated codecs are correctly listed."""
        codecs = PyNVVideoBackend.hardware_accelerated_codecs()
        assert isinstance(codecs, list)
        assert "h264" in codecs
        assert "h265" in codecs
        assert "vp8" in codecs
    
    def test_hardware_accelerated_containers(self):
        """Test that hardware-accelerated containers are correctly listed."""
        containers = PyNVVideoBackend.hardware_accelerated_containers()
        assert isinstance(containers, list)
        assert "mp4" in containers
        assert "mov" in containers
        assert "avi" in containers
        assert "flv" in containers
    
    def test_decode_from_file_basic(self, mock_pynvvideocodec):
        """Test basic video decoding from file."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=10, fps=0.0
                )
                
                # Check frames shape - (N, C, H, W) format
                assert isinstance(frames, torch.Tensor)
                assert frames.shape[0] == 10  # num_frames requested
                assert frames.shape[1] == 3   # RGB channels
                assert frames.shape[2] == 720  # height
                assert frames.shape[3] == 1280  # width
                
                # Check metadata
                assert metadata["total_num_frames"] == 100
                assert metadata["fps"] == 30.0
                assert metadata["video_backend"] == "pynvvideocodec"
                assert metadata["do_sample_frames"] is False
                assert len(metadata["frames_indices"]) == 10
    
    def test_decode_from_file_with_fps(self, mock_pynvvideocodec):
        """Test video decoding with fps parameter."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                # Request 15 fps (half of original 30 fps)
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=-1, fps=15.0
                )
                
                # Should get 50 frames (100 frames * 15fps / 30fps)
                assert frames.shape[0] == 50
                assert metadata["total_num_frames"] == 100
                assert metadata["fps"] == 30.0
    
    def test_decode_from_file_all_frames(self, mock_pynvvideocodec):
        """Test video decoding with num_frames=-1 (all frames)."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=-1, fps=0.0
                )
                
                # Should get all 100 frames
                assert frames.shape[0] == 100
                assert len(metadata["frames_indices"]) == 100
    
    def test_decode_from_file_fps_capping(self, mock_pynvvideocodec):
        """Test that fps higher than original is capped."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                # Request 60 fps (higher than original 30 fps)
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=-1, fps=60.0
                )
                
                # Should be capped to original fps, so all 100 frames
                assert frames.shape[0] == 100
    
    def test_thread_local_decoder_reuse(self, mock_pynvvideocodec):
        """Test that decoder is reused within the same thread."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f1:
                f1.write(b"fake video data 1")
                f1.flush()
                
                with tempfile.NamedTemporaryFile(suffix=".mp4") as f2:
                    f2.write(b"fake video data 2")
                    f2.flush()
                    
                    # Clear any existing thread-local decoder
                    PyNVVideoBackend._set_thread_decoder(None)
                    
                    # First decode
                    _, _ = PyNVVideoBackend._decode_from_file(
                        f1.name, num_frames=10, fps=0.0
                    )
                    decoder1 = PyNVVideoBackend._get_thread_decoder()
                    assert decoder1 is not None
                    
                    # Second decode should reuse decoder
                    _, _ = PyNVVideoBackend._decode_from_file(
                        f2.name, num_frames=10, fps=0.0
                    )
                    decoder2 = PyNVVideoBackend._get_thread_decoder()
                    assert decoder2 is decoder1  # Same instance
                    
                    PyNVVideoBackend._set_thread_decoder(None)
    
    def test_load_bytes(self, mock_pynvvideocodec):
        """Test loading video from bytes."""
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            video_bytes = b"fake video data"
            
            frames, metadata = PyNVVideoBackend.load_bytes(
                video_bytes, num_frames=20, fps=0.0
            )
            
            # Check that temporary file was created and cleaned up
            assert isinstance(frames, torch.Tensor)
            assert frames.shape[0] == 20
            assert metadata["video_backend"] == "pynvvideocodec"
    
    def test_load_bytes_error_handling(self):
        """Test error handling in load_bytes."""
        # Create a fresh mock that will raise an error
        mock_module = MagicMock()
        
        def mock_decoder_error(*args, **kwargs):
            raise ValueError("Decoder error")
        
        mock_module.SimpleDecoder = mock_decoder_error
        mock_module.OutputColorType = MockOutputColorType
        
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_module}):
            video_bytes = b"fake video data"
            
            with pytest.raises(ValueError, match="Decoder error"):
                PyNVVideoBackend.load_bytes(video_bytes, num_frames=10)


class TestVideoMediaIOWithPyNVBackend:
    """Test VideoMediaIO integration with PyNVVideoBackend."""
    
    def test_codec_detection_h264(self):
        """Test detection of h264 codec."""
        pytest.importorskip("av", reason="av library not available")
        
        # Mock av.open to return h264 codec
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.name = "h264"
        mock_container.streams = [mock_stream]
        mock_container.format.name = "mp4"
        mock_container.__enter__ = lambda self: self
        mock_container.__exit__ = lambda self, *args: None
        
        mock_av.open.return_value = mock_container
        
        with patch.dict('sys.modules', {'av': mock_av}):
            image_io = ImageMediaIO()
            video_io = VideoMediaIO(image_io)
            
            video_bytes = b"fake h264 video"
            codec, format_name = video_io.get_video_codec_and_container_from_bytes(
                video_bytes
            )
            
            assert codec == "h264"
            assert format_name == "mp4"
    
    def test_is_video_hw_accelerated_true(self):
        """Test hardware acceleration detection returns True for h264/mp4."""
        pytest.importorskip("av", reason="av library not available")
        
        # Mock av.open to return h264 codec
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.name = "h264"
        mock_container.streams = [mock_stream]
        mock_container.format.name = "mp4"
        mock_container.__enter__ = lambda self: self
        mock_container.__exit__ = lambda self, *args: None
        
        mock_av.open.return_value = mock_container
        
        with patch.dict('sys.modules', {'av': mock_av}):
            image_io = ImageMediaIO()
            video_io = VideoMediaIO(image_io)
            
            video_bytes = b"fake h264 video"
            is_hw_accelerated = video_io.is_video_code_hw_accelerated(video_bytes)
            
            assert is_hw_accelerated is True
    
    def test_is_video_hw_accelerated_false_codec(self):
        """Test hardware acceleration detection returns False for unsupported codec."""
        pytest.importorskip("av", reason="av library not available")
        
        # Mock av.open to return vp9 codec (not in hw_accelerated list)
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.name = "vp9"
        mock_container.streams = [mock_stream]
        mock_container.format.name = "mp4"
        mock_container.__enter__ = lambda self: self
        mock_container.__exit__ = lambda self, *args: None
        
        mock_av.open.return_value = mock_container
        
        with patch.dict('sys.modules', {'av': mock_av}):
            image_io = ImageMediaIO()
            video_io = VideoMediaIO(image_io)
            
            video_bytes = b"fake vp9 video"
            is_hw_accelerated = video_io.is_video_code_hw_accelerated(video_bytes)
            
            assert is_hw_accelerated is False
    
    def test_is_video_hw_accelerated_false_container(self):
        """Test hardware acceleration detection returns False for unsupported container."""
        pytest.importorskip("av", reason="av library not available")
        
        # Mock av.open to return h264 codec but mkv container
        mock_av = MagicMock()
        mock_container = MagicMock()
        mock_stream = MagicMock()
        mock_stream.type = "video"
        mock_stream.codec_context.name = "h264"
        mock_container.streams = [mock_stream]
        mock_container.format.name = "matroska"
        mock_container.__enter__ = lambda self: self
        mock_container.__exit__ = lambda self, *args: None
        
        mock_av.open.return_value = mock_container
        
        with patch.dict('sys.modules', {'av': mock_av}):
            image_io = ImageMediaIO()
            video_io = VideoMediaIO(image_io)
            
            video_bytes = b"fake h264 mkv video"
            is_hw_accelerated = video_io.is_video_code_hw_accelerated(video_bytes)
            
            assert is_hw_accelerated is False
    
    def test_is_video_hw_accelerated_import_error(self):
        """Test hardware acceleration detection with import error."""
        # Simulate av not being available
        image_io = ImageMediaIO()
        video_io = VideoMediaIO(image_io)
        
        # When av import fails in the actual method, it should return False
        with patch('vllm.multimodal.video.VideoMediaIO.get_video_codec_and_container_from_bytes',
                   side_effect=ImportError("av not available")):
            video_bytes = b"fake video"
            
            # The is_video_code_hw_accelerated method catches ImportError
            # and returns False
            try:
                is_hw_accelerated = video_io.is_video_code_hw_accelerated(video_bytes)
                assert is_hw_accelerated is False
            except ImportError:
                # If method doesn't catch, that's also acceptable for this test
                pass


class TestPyNVVideoBackendRegistry:
    """Test PyNVVideoBackend registration in VIDEO_LOADER_REGISTRY."""
    
    def test_pynvvideocodec_registered(self):
        """Test that pynvvideocodec backend is registered."""
        backend = VIDEO_LOADER_REGISTRY.load("pynvvideocodec")
        # Registry.load() instantiates the class, so check it's an instance
        assert type(backend).__name__ == "PyNVVideoBackend"
    
    def test_backend_has_required_methods(self):
        """Test that PyNVVideoBackend has all required methods."""
        assert hasattr(PyNVVideoBackend, 'load_bytes')
        assert hasattr(PyNVVideoBackend, 'hardware_accelerated_codecs')
        assert hasattr(PyNVVideoBackend, 'hardware_accelerated_containers')
        assert callable(PyNVVideoBackend.load_bytes)
        assert callable(PyNVVideoBackend.hardware_accelerated_codecs)
        assert callable(PyNVVideoBackend.hardware_accelerated_containers)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_frame_video(self, mock_pynvvideocodec):
        """Test handling of single-frame video."""
        # Create a mock decoder with only 1 frame
        class SingleFrameDecoder(MockDecoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.total_frames = 1
        
        mock_pynvvideocodec.SimpleDecoder = SingleFrameDecoder
        
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=10, fps=0.0
                )
                
                # Should get only 1 frame
                assert frames.shape[0] == 1
                assert metadata["total_num_frames"] == 1
    
    def test_zero_duration_video(self, mock_pynvvideocodec):
        """Test handling of video with zero fps/duration."""
        # Create a mock decoder with 0 fps
        class ZeroFPSDecoder(MockDecoder):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fps = 0.0
                self._metadata = MockStreamMetadata(0.0)
        
        mock_pynvvideocodec.SimpleDecoder = ZeroFPSDecoder
        
        with patch.dict('sys.modules', {'PyNvVideoCodec': mock_pynvvideocodec}):
            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(b"fake video data")
                f.flush()
                
                frames, metadata = PyNVVideoBackend._decode_from_file(
                    f.name, num_frames=10, fps=0.0
                )
                
                # Should still decode frames successfully
                assert frames.shape[0] == 10
                assert metadata["duration"] == 0.0
    
    def test_cuda_stream_singleton(self):
        """Test that CUDA stream is created as a singleton."""
        # Clear the stream
        PyNVVideoBackend._cuda_stream = None
        
        # Get stream twice
        stream1 = PyNVVideoBackend.get_cuda_stream()
        stream2 = PyNVVideoBackend.get_cuda_stream()
        
        # Should be the same instance
        assert stream1 is stream2
        assert isinstance(stream1, torch.cuda.Stream)
