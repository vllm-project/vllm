// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Energy-based splitting of long audio into model-sized clips.
//!
//! Original Python definition:
//! <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/multimodal/audio.py#L430-L555>

/// How long audio is split before transcription.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChunkingConfig {
    /// Maximum length of a clip, in samples.
    pub max_clip_samples: usize,
    /// Length of the region before each hard boundary that is searched for a
    /// quiet split point, in samples.
    pub search_samples: usize,
    /// Size of the windows whose RMS energy is compared, in samples.
    pub energy_window_samples: usize,
}

/// Split mono audio into clips of at most `max_clip_samples` samples,
/// splitting long audio at low-energy regions to minimize cutting through
/// speech. Returns the sample range of each clip.
pub(crate) fn split_audio(audio: &[f32], config: ChunkingConfig) -> Vec<std::ops::Range<usize>> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < audio.len() {
        if start + config.max_clip_samples >= audio.len() {
            // Handle last chunk - take everything remaining
            chunks.push(start..audio.len());
            break;
        }

        // Find the best split point in the overlap region
        let search_start = start + config.max_clip_samples - config.search_samples;
        let search_end = start + config.max_clip_samples;
        let mut split_point = find_split_point(
            audio,
            search_start,
            search_end,
            config.energy_window_samples,
        );
        // Guarantee forward progress: if split_point didn't advance,
        // fall back to the hard chunk boundary.
        if split_point <= start {
            split_point = search_end;
        }

        chunks.push(start..split_point);
        start = split_point;
    }
    chunks
}

/// Find the best point to split audio by looking for silence or low
/// amplitude: the start of the window with the lowest RMS energy in
/// `start..end`.
fn find_split_point(audio: &[f32], start: usize, end: usize, window: usize) -> usize {
    let segment = &audio[start..end];
    let mut min_energy = f64::INFINITY;
    let mut quietest = start;
    // Like Python's `range(0, len(segment) - window, window)`, skip the last
    // window when it ends exactly at `end`.
    let mut offset = 0;
    while window > 0 && offset + window < segment.len() {
        let samples = &segment[offset..offset + window];
        let mean_square =
            samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / window as f64;
        let energy = mean_square.sqrt();
        if energy < min_energy {
            quietest = start + offset;
            min_energy = energy;
        }
        offset += window;
    }
    quietest
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: ChunkingConfig = ChunkingConfig {
        max_clip_samples: 30 * 16_000,
        search_samples: 16_000,
        energy_window_samples: 1_600,
    };

    #[test]
    fn audio_up_to_the_clip_length_is_one_clip() {
        let chunks = split_audio(&[0.5; 30 * 16_000], CONFIG);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], 0..30 * 16_000);
        assert!(split_audio(&[], CONFIG).is_empty());
    }

    #[test]
    fn long_audio_splits_at_the_quietest_window_before_each_boundary() {
        // 65 s of constant tone with one silent window 0.5 s before the first
        // 30 s boundary.
        let mut audio = vec![0.5_f32; 1_040_000];
        audio[472_000..473_600].fill(0.0);

        // The first clip ends at the silence. The second finds no quieter
        // window, so it ends at the start of its search region, 1 s before
        // its hard boundary. The rest is the last clip.
        assert_eq!(
            split_audio(&audio, CONFIG),
            [0..472_000, 472_000..936_000, 936_000..1_040_000]
        );
    }
}
