# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF
from torchmetrics.functional.image import structural_similarity_index_measure


class SimilarFrameDetector:
    def __init__(
        self,
        sparse_ratio: float = 0.5,
        use_downsampled_loss: bool = True,
        downscale_factor: int = 4,
        alpha: float = 0.85,
    ):
        self.sparse_ratio = sparse_ratio
        self.use_downsampled_loss = use_downsampled_loss
        self.downscale_factor = downscale_factor
        self.alpha = alpha

    def _detect_and_convert_format(
        self, video_data: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, str]:
        """Convert input to channels_first tensor and record original format."""
        if isinstance(video_data, np.ndarray):
            if video_data.ndim != 4:
                raise ValueError("Input must be 4-dimensional array")
            video_tensor = torch.from_numpy(video_data)
        else:
            video_tensor = video_data.clone()

        if video_tensor.ndim != 4:
            raise ValueError("Input must be 4-dimensional tensor")

        if video_tensor.shape[1] == 3:
            converted_data = video_tensor
            original_format = "channels_first"
        elif video_tensor.shape[-1] == 3:
            converted_data = video_tensor.permute(0, 3, 1, 2)
            original_format = "channels_last"
        else:
            raise ValueError("Input must have 3 channels (RGB)")

        return converted_data, original_format

    def _convert_back_to_original_format(
        self, video_data: torch.Tensor, original_format: str
    ) -> torch.Tensor:
        """Convert tensor back to original channel format."""
        if original_format == "channels_last" and video_data.numel() > 0:
            return video_data.permute(0, 2, 3, 1)
        return video_data

    def _calculate_target_frames(self, total_frames: int) -> int:
        """Calculate target keyframe count (even, ≥2, ≤total_frames)."""
        k = int(total_frames * self.sparse_ratio)
        k = max(2, min(total_frames, 2 * (k // 2)))
        return k

    def _downsample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Downsample frames to reduce computation cost."""
        frame_number, channels, height, width = frames.shape
        new_height, new_width = (
            height // self.downscale_factor,
            width // self.downscale_factor,
        )

        downsampled = tvF.resize(
            frames,
            size=(new_height, new_width),
            interpolation=tvF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return downsampled

    def _calculate_ssim(self, gray1: torch.Tensor, gray2: torch.Tensor) -> torch.Tensor:
        """Calculate SSIM for batch of frames."""
        ssim_values = structural_similarity_index_measure(
            preds=gray1,
            target=gray2,
            kernel_size=11,
            data_range=1.0,
            gaussian_kernel=True,
            reduction="none",
        )

        return ssim_values.float()

    def _calculate_photometric_loss(
        self, frames1: torch.Tensor, frames2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate photometric loss (weighted SSIM + L1) for batch of frames."""
        assert frames1.shape == frames2.shape
        assert frames1.device == frames2.device
        assert frames1.dim() == 4, "Input must be 4D tensor (B, C, H, W)"
        assert frames1.shape[1] == 1, "Input must be grayscale (1 channel)"

        ssim_values = self._calculate_ssim(frames1, frames2)

        l1_losses = F.l1_loss(frames1.float(), frames2.float(), reduction="none")
        l1_losses = l1_losses.mean(dim=[1, 2, 3])

        ssim_losses = (1 - ssim_values) / 2
        photometric_losses = self.alpha * ssim_losses + (1 - self.alpha) * l1_losses

        return photometric_losses

    def _calculate_video_photometric_losses(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute photometric loss for all adjacent frame pairs in batch."""
        frame_number = frames.shape[0]
        if frame_number < 2:
            return torch.tensor([], device=frames.device)

        frames_gray = tvF.rgb_to_grayscale(frames, num_output_channels=1)
        frames_gray = tvF.convert_image_dtype(frames_gray, dtype=torch.float32)

        prev_frames = frames_gray[:-1]
        next_frames = frames_gray[1:]

        return self._calculate_photometric_loss(prev_frames, next_frames)

    def _select_split_points(
        self, photometric_losses: torch.Tensor, k: int
    ) -> list[int]:
        """Select top-k largest loss indices as split points."""
        if k <= 1:
            return []
        top_indices = torch.topk(photometric_losses, k - 1).indices.tolist()
        return sorted(top_indices)

    def _create_segments(
        self, split_points: list[int], total_frames: int
    ) -> list[tuple[int, int]]:
        """Split video frames into segments via split points."""
        segments = []
        start = 0
        for split_point in split_points:
            end = split_point + 1
            if end > start:
                segments.append((start, end))
            start = end

        if start < total_frames:
            segments.append((start, total_frames))
        return segments

    def _select_keyframes_from_segments(
        self, video_data: torch.Tensor, segments: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, list[int]]:
        """Select middle frame of each segment as keyframe."""
        selected_frames = []
        selected_frame_indices = []

        for seg_start, seg_end in segments:
            seg_length = seg_end - seg_start
            if seg_length > 0:
                mid_index = seg_start + seg_length // 2
                if mid_index < video_data.shape[0]:
                    selected_frames.append(video_data[mid_index])
                    selected_frame_indices.append(mid_index)

        if selected_frames:
            return torch.stack(selected_frames, dim=0), selected_frame_indices
        else:
            fallback_frames = [video_data[0], video_data[-1]]
            fallback_indices = [0, video_data.shape[0] - 1]
            return torch.stack(fallback_frames, dim=0), fallback_indices

    def preprocess(
        self, videos: list[torch.Tensor | tuple]
    ) -> tuple[list[torch.Tensor], bool]:
        """Preprocess input video list (extract tensor from tuple if needed)."""
        processed = False
        if not videos:
            return [], processed

        first_element = videos[0]

        if (
            torch.is_tensor(first_element)
            or isinstance(first_element, np.ndarray)
            and len(first_element) >= 1
        ):
            return videos, processed
        elif (
            isinstance(first_element, tuple)
            and len(first_element) >= 1
            and torch.is_tensor(first_element[0])
            or isinstance(first_element, tuple)
            and len(first_element) >= 1
            and isinstance(first_element[0], np.ndarray)
        ):
            videos_t = [item[0] for item in videos]
            processed = True
            return videos_t, processed
        else:
            raise ValueError("Unsupported input format.")

    def frame_sampling(
        self, video_list: list[np.ndarray | torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """Core method: sample keyframes from video list."""
        if not isinstance(video_list, list):
            raise TypeError("Input must be a list")

        if len(video_list) == 0:
            raise ValueError("Input list cannot be empty")

        original_formats = []
        original_types = []
        processed_video_list = []

        for i, video_data in enumerate(video_list):
            converted_data, original_format = self._detect_and_convert_format(
                video_data
            )
            original_formats.append(original_format)
            original_types.append(type(video_data))
            processed_video_list.append(converted_data)

        result_videos = []
        result_selected_frames_index = []

        for video_idx, video_data in enumerate(processed_video_list):
            frame_number, channels, height, width = video_data.shape

            if frame_number == 1:
                selected_frames = video_data
                selected_frames_index = [0]
                result_selected_frames_index.append(selected_frames_index)
                result_videos.append(selected_frames)
                continue

            k = self._calculate_target_frames(frame_number)

            working_frames = (
                self._downsample_frames(video_data)
                if self.use_downsampled_loss
                else video_data
            )

            video_losses = self._calculate_video_photometric_losses(working_frames)
            split_points = self._select_split_points(video_losses, k)
            segments = self._create_segments(split_points, frame_number)
            selected_frames, selected_frames_index = (
                self._select_keyframes_from_segments(video_data, segments)
            )
            result_selected_frames_index.append(selected_frames_index)
            result_videos.append(selected_frames)

        final_results = []
        for i, (result, original_format, original_type) in enumerate(
            zip(result_videos, original_formats, original_types)
        ):
            converted_result = self._convert_back_to_original_format(
                result, original_format
            )
            if original_type == np.ndarray:
                converted_result = converted_result.cpu().numpy()
            final_results.append(converted_result)

        return final_results, result_selected_frames_index

    def process_video_frames(
        self, videos: list[torch.Tensor | tuple]
    ) -> list[torch.Tensor] | list[tuple]:
        """End-to-end video frame sampling (preprocess + sampling + metadata update)."""
        video_list, processed = self.preprocess(videos)
        video_sampled, sampled_frames_index = self.frame_sampling(video_list)

        if processed:
            result = []
            for i, (_, *metadata) in enumerate(videos):
                video_metadata = metadata[0] if metadata else {}

                frame_number = video_list[i].shape[0]
                if frame_number == 1:
                    updated_metadata = video_metadata.copy()
                    updated_metadata["frames_indices"] = [0]
                    result.append((video_sampled[i], updated_metadata))
                    continue

                frames_indices = video_metadata.get("frames_indices", torch.tensor([]))
                if i < len(sampled_frames_index) and len(frames_indices) > 0:
                    if torch.is_tensor(frames_indices):
                        selected_indices = torch.tensor(
                            [
                                frames_indices[idx].item()
                                for idx in sampled_frames_index[i]
                                if idx < len(frames_indices)
                            ]
                        )
                    else:
                        selected_indices = [
                            frames_indices[idx]
                            for idx in sampled_frames_index[i]
                            if idx < len(frames_indices)
                        ]
                else:
                    selected_indices = frames_indices

                updated_metadata = video_metadata.copy()
                updated_metadata["frames_indices"] = selected_indices
                result.append((video_sampled[i], updated_metadata))
            return result
        else:
            return video_sampled


def is_multimodal_efs_enabled(efs_sparse_rate: float | None) -> bool:
    """Check if EFS (Efficient Frame Sampling) is enabled."""
    return efs_sparse_rate is not None and efs_sparse_rate > 0
