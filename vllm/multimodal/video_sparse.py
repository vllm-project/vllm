# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.ops import structural_similarity_index_measure as ssim

class SimilarFrameDetector:
    """
    Detects similar frames in video and samples keyframes
    based on photometric loss (SSIM + L1).
    Reduces redundant frames by selecting representative
    keyframes at a specified sparse ratio.
    """

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
        """
        Convert input video data to unified tensor
        format (channels_first) and record original format.

        Args:
            video_data: Input video (4D: [frames, C, H, W] or [frames, H, W, C])

        Returns:
            Converted tensor (channels_first), original format label
        """
        if isinstance(video_data, np.ndarray):
            if video_data.ndim != 4:
                raise ValueError("Input must be 4-dimensional array")
            video_tensor = torch.from_numpy(video_data)
        else:
            video_tensor = video_data.clone()

        if video_tensor.ndim != 4:
            raise ValueError("Input must be 4-dimensional tensor")

        # Convert to channels_first format
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
        """Convert tensor back to original channel format (channels_first/last)."""
        if original_format == "channels_last" and video_data.numel() > 0:
            return video_data.permute(0, 2, 3, 1)
        else:
            return video_data

    def _calculate_target_frames(self, total_frames: int) -> int:
        """Calculate target number of keyframes to sample (ensure even number ≥2)."""
        k = int(total_frames * self.sparse_ratio)
        # Ensure k is even and between 2 and total_frames
        k = max(2, min(total_frames, 2 * (k // 2)))
        return k

    def _downsample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Downsample frames to reduce computation cost.
        """
        frame_number, channels, height, width = frames.shape
        new_height, new_width = (
            height // self.downscale_factor,
            width // self.downscale_factor,
        )

        downsampled = tvF.resize(
            img=frames,
            size=(new_height, new_width),
            interpolation=tvF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return downsampled

    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two frames.
        Higher SSIM means more similar frames (range: 0-1).
        """
        assert img1.shape == img2.shape
        assert img1.device == img2.device

        if img1.shape[0] == 3:
            gray1 = 0.299 * img1[0] + 0.587 * img1[1] + 0.114 * img1[2]
            gray2 = 0.299 * img2[0] + 0.587 * img2[1] + 0.114 * img2[2]
        else:
            gray1 = img1[0]
            gray2 = img2[0]

        gray1 = gray1.unsqueeze(0).unsqueeze(0)
        gray2 = gray2.unsqueeze(0).unsqueeze(0)

        ssim_value = ssim(gray1, gray2, win_size=11, sigma=1.5, data_range=255.0, size_average=True
        )

        return float(ssim_value)

    def _calculate_photometric_loss(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> float:
        """
        Calculate photometric loss (weighted SSIM + L1 loss) between two frames.
        Lower loss means more similar frames.
        """
        ssim = self._calculate_ssim(frame1, frame2)
        # Compute L1 loss (pixel-wise absolute difference)
        if frame1.shape[0] == 3:
            l1_loss = torch.mean(torch.abs(frame1 - frame2).float()).item()
        else:
            l1_loss = torch.mean(torch.abs(frame1[0] - frame2[0])).item()
        # Weighted combination of SSIM and L1 loss
        loss = self.alpha * (1 - ssim) / 2 + (1 - self.alpha) * l1_loss
        return loss

    def _calculate_video_photometric_losses(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute photometric loss for all adjacent frame pairs in video."""
        frame_number = frames.shape[0]
        losses = []
        for f in range(frame_number - 1):
            loss = self._calculate_photometric_loss(frames[f], frames[f + 1])
            losses.append(loss)
        return torch.tensor(losses, device=frames.device)

    def _select_split_points(
        self, photometric_losses: torch.Tensor, k: int
    ) -> list[int]:
        """
        Select split points by top-k largest loss
        values (frame pairs with biggest changes).
        These points divide video into k segments.
        """
        if k - 1 <= 0:
            return []
        # Select top (k-1) loss indices as split points
        top_indices = torch.topk(photometric_losses, k - 1).indices.tolist()
        split_points = sorted(top_indices)
        return split_points

    def _create_segments(
        self, split_points: list[int], total_frames: int
    ) -> list[tuple[int, int]]:
        """Split video frame indices into segments based on split points."""
        segments = []
        start = 0
        for split_point in split_points:
            end = split_point + 1
            if end > start:
                segments.append((start, end))
            start = end

        # Add the last segment
        if start < total_frames:
            segments.append((start, total_frames))
        return segments

    def _select_keyframes_from_segments(
        self, video_data: torch.Tensor, segments: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Select middle frame of each segment as keyframe (representative of the segment).
        Fallback to first/last frame if no valid segments.
        """
        selected_frames = []
        selected_frame_indices = []

        for seg_start, seg_end in segments:
            seg_length = seg_end - seg_start
            if seg_length > 0:
                mid_index = seg_start + seg_length // 2
                if mid_index < video_data.shape[0]:
                    selected_frames.append(video_data[mid_index])
                    selected_frame_indices.append(mid_index)

        # Fallback: select first and last frame if no valid keyframes
        if selected_frames:
            return torch.stack(selected_frames, dim=0), selected_frame_indices
        else:
            fallback_frames = [video_data[0], video_data[-1]]
            fallback_indices = [0, video_data.shape[0] - 1]
            return torch.stack(fallback_frames, dim=0), fallback_indices

    def preprocess(
        self, videos: list[torch.Tensor | tuple]
    ) -> tuple[list[torch.Tensor], bool]:
        """
        Preprocess input video list: extract tensor data from tuple (if needed).

        Args:
            videos: List of tensors or (tensor + metadata) tuples

        Returns:
            List of video tensors, flag indicating if input was tuple-based
        """
        processed = False
        if not videos:
            return [], processed

        first_element = videos[0]

        if torch.is_tensor(first_element):
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
            raise ValueError("unsupported input format.")

    def frame_sampling(
        self, video_list: list[np.ndarray | torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[list[int]]]:
        """
        Core method: sample keyframes from video list based on photometric loss.

        Args:
            video_list: List of video data (numpy array/tensor)

        Returns:
            Sampled keyframes, original indices of sampled frames
        """
        if not isinstance(video_list, list):
            raise TypeError("Input must be a list")

        if len(video_list) == 0:
            raise ValueError("Input list cannot be empty")

        original_formats = []
        original_types = []
        processed_video_list = []

        # Unify input format for all videos
        for i, video_data in enumerate(video_list):
            converted_data, original_format = self._detect_and_convert_format(
                video_data
            )
            original_formats.append(original_format)
            original_types.append(type(video_data))
            processed_video_list.append(converted_data)

        result_videos = []
        result_selected_frames_index = []

        # Sample keyframes for each video
        for video_idx, video_data in enumerate(processed_video_list):
            frame_number, channels, height, width = video_data.shape

            if frame_number == 1:
                selected_frames = video_data
                selected_frames_index = [0]
                result_selected_frames_index.append(selected_frames_index)
                result_videos.append(selected_frames)
                continue

            k = self._calculate_target_frames(frame_number)

            # Use downsampled frames for loss calculation (speed up)
            working_frames = (
                self._downsample_frames(video_data)
                if self.use_downsampled_loss
                else video_data
            )

            # Compute loss, split segments, select keyframes
            video_losses = self._calculate_video_photometric_losses(working_frames)
            split_points = self._select_split_points(video_losses, k)
            segments = self._create_segments(split_points, frame_number)
            selected_frames, selected_frames_index = (
                self._select_keyframes_from_segments(video_data, segments)
            )
            result_selected_frames_index.append(selected_frames_index)
            result_videos.append(selected_frames)

        # Convert back to original format/type
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
        """
        End-to-end video frame sampling (preprocess + frame sampling + metadata update).

        Args:
            videos: List of tensors or (tensor + metadata) tuples

        Returns:
            Sampled keyframes (with updated metadata if input was tuple)
        """
        video_list, processed = self.preprocess(videos)
        video_sampled, sampled_frames_index = self.frame_sampling(video_list)

        # Update metadata (e.g., frame indices) if input was tuple-based
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
                # Update frame indices to sampled keyframes
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
    """Check if EFS (Efficient Frame Sampling) is enabled (valid sparse rate > 0)."""
    return efs_sparse_rate is not None and efs_sparse_rate > 0
