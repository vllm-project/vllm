// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! HY4-VL video frame sampling: the reference `sample_frames` index rule.
//!
//! Port of `HYV4VLVideoProcessor.sample_frames` in
//! `transformers_utils/processors/hy_v4.py` (the `fps` branch; the mutually
//! exclusive `num_frames` branch is not ported — serving samples by rate).
//! This is the index rule the media connector will be driven with through
//! the upcoming `fetch_video_frames(source, cfg, indices)` integration in
//! llm-multimodal; until that lands the connector keeps the spec-declared
//! `FrameSampling::EvenRounded` and this module has no caller outside tests.

use llm_multimodal::vision::transforms::round_half_to_even;

/// The picked source-frame indices of the reference `sample_frames`:
/// `count = min(total, max(min_frames, min(max_frames, floor(total /
/// container_fps * sample_fps))))` frames — a non-positive `sample_fps`
/// takes the reference's `else maximum` branch — then each segment's
/// midpoint `min((round(segment * i) + round(segment * (i + 1))) // 2,
/// total - 1)` with `segment = (total - 1) / count`, where `round` is
/// Python's round-half-to-even (`np.round`) and `//` integer floor
/// division. Assumes `min_frames >= 1` like the reference's
/// `kwargs.get("min_frames") or self.min_frames` defaulting.
///
/// Returns `None` where the reference raises `ValueError`: no frames in the
/// container or a non-positive/non-finite container fps.
// TODO(fetch_video_frames): dead until the connector integration lands.
#[allow(dead_code)]
pub(super) fn sample_frame_indices(
    total_frames: usize,
    container_fps: f64,
    sample_fps: f64,
    min_frames: usize,
    max_frames: usize,
) -> Option<Vec<usize>> {
    if total_frames == 0 || !(container_fps.is_finite() && container_fps > 0.0) {
        return None;
    }
    let desired = if sample_fps > 0.0 {
        (total_frames as f64 / container_fps * sample_fps).floor() as usize
    } else {
        max_frames
    };
    let count = total_frames.min(min_frames.max(max_frames.min(desired)));
    let segment = (total_frames - 1) as f64 / count as f64;
    Some(
        (0..count)
            .map(|i| {
                let lo = round_half_to_even(segment * i as f64) as usize;
                let hi = round_half_to_even(segment * (i + 1) as f64) as usize;
                ((lo + hi) / 2).min(total_frames - 1)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expected index list in this module was produced by running the
    /// reference `sample_frames` formula (`np.round` included) verbatim.
    #[test]
    fn picks_segment_midpoints_at_the_reference_rate() {
        // 240 frames at 24 fps sampled at 2 fps: floor(240/24*2) = 20
        // picks, segment 239/20 = 11.95, midpoints 6, 18, 30, ...
        assert_eq!(
            sample_frame_indices(240, 24.0, 2.0, 4, 100).unwrap(),
            vec![
                6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 125, 137, 149, 161, 173, 185, 197, 209,
                221, 233
            ]
        );
        // A non-integer desired count floors: floor(240/24*1.5) = 15.
        assert_eq!(
            sample_frame_indices(240, 24.0, 1.5, 4, 100).unwrap(),
            vec![
                8, 24, 40, 56, 72, 88, 104, 119, 135, 151, 167, 183, 199, 215, 231
            ]
        );
        // A fractional container fps: floor(7/29.97*2) = 0 -> min_frames 4.
        assert_eq!(
            sample_frame_indices(7, 29.97, 2.0, 4, 100).unwrap(),
            vec![1, 2, 3, 5]
        );
    }

    #[test]
    fn segment_rounding_is_python_half_to_even() {
        // Segment 5/2 = 2.5: np.round(2.5) = 2 (half to even), so the picks
        // are (0 + 2) // 2 = 1 and (2 + 5) // 2 = 3; round-half-up would
        // give [1, 4].
        assert_eq!(
            sample_frame_indices(6, 24.0, 2.0, 2, 100).unwrap(),
            vec![1, 3]
        );
        // Segment 1/2 = 0.5: np.round(0.5) = 0, so both picks land on
        // frame 0; round-half-up would give [0, 1].
        assert_eq!(
            sample_frame_indices(2, 2.0, 2.0, 1, 100).unwrap(),
            vec![0, 0]
        );
        // Segment 3/2 = 1.5: np.round(1.5) = 2.
        assert_eq!(
            sample_frame_indices(4, 24.0, 2.0, 2, 100).unwrap(),
            vec![1, 2]
        );
        // Segment 10/4 = 2.5 again, reached via the min_frames clamp.
        assert_eq!(
            sample_frame_indices(11, 24.0, 2.0, 4, 100).unwrap(),
            vec![1, 3, 6, 9]
        );
    }

    #[test]
    fn count_clamps_to_bounds_and_clip_length() {
        // floor(10000/24*2) = 833 exceeds max_frames: capped at 100.
        let indices = sample_frame_indices(10000, 24.0, 2.0, 4, 100).unwrap();
        assert_eq!(indices.len(), 100);
        assert_eq!(indices[0], 50);
        assert_eq!(indices[49], 4950);
        assert_eq!(indices[99], 9949);
        // floor(8/24*2) = 0 is floored at min_frames.
        assert_eq!(
            sample_frame_indices(8, 24.0, 2.0, 4, 100).unwrap(),
            vec![1, 3, 4, 6]
        );
        // A clip shorter than min_frames yields min(total, min_frames)
        // picks, repeating the tail frame via the total - 1 clamp.
        assert_eq!(
            sample_frame_indices(3, 24.0, 2.0, 4, 100).unwrap(),
            vec![0, 1, 1]
        );
        // A non-positive sample rate takes the reference's `else maximum`
        // branch.
        assert_eq!(
            sample_frame_indices(240, 24.0, 0.0, 4, 100).unwrap().len(),
            100
        );
    }

    #[test]
    fn invalid_metadata_yields_no_indices() {
        assert_eq!(sample_frame_indices(0, 24.0, 2.0, 4, 100), None);
        assert_eq!(sample_frame_indices(240, 0.0, 2.0, 4, 100), None);
        assert_eq!(sample_frame_indices(240, -1.0, 2.0, 4, 100), None);
        assert_eq!(sample_frame_indices(240, f64::NAN, 2.0, 4, 100), None);
    }
}
