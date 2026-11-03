// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Audio resampling for the MuseSpark frontend.
//!
//! Ported from `llm-multimodal`'s `audio::transforms`, which is `pub(crate)`
//! there and cannot be imported.

use std::f32::consts::PI as PI_F32;

use llm_multimodal::TransformError;

/// Match torchaudio's default `functional.resample`: band-limited sinc
/// interpolation with a Hann window, filter width 6, and rolloff 0.99.
pub(super) fn bandlimited_resample(
    samples: &[f32],
    src_sample_rate: usize,
    dst_sample_rate: usize,
) -> Result<Vec<f32>, TransformError> {
    const LOWPASS_FILTER_WIDTH: f32 = 6.0;
    const ROLLOFF: f32 = 0.99;

    if src_sample_rate == 0 || dst_sample_rate == 0 {
        return Err(TransformError::ShapeError(
            "audio resampling rates must be positive".to_string(),
        ));
    }
    if samples.is_empty() || src_sample_rate == dst_sample_rate {
        return Ok(samples.to_vec());
    }

    let gcd = greatest_common_divisor(src_sample_rate, dst_sample_rate);
    let orig_freq = src_sample_rate / gcd;
    let new_freq = dst_sample_rate / gcd;
    let base_freq = orig_freq.min(new_freq) as f32 * ROLLOFF;
    let width = (LOWPASS_FILTER_WIDTH * orig_freq as f32 / base_freq).ceil() as usize;
    let kernel_len = width
        .checked_mul(2)
        .and_then(|value| value.checked_add(orig_freq))
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel is too large".into()))?;
    let kernel_values = new_freq
        .checked_mul(kernel_len)
        .ok_or_else(|| TransformError::ShapeError("audio resample kernel size overflow".into()))?;
    let mut kernels = Vec::new();
    kernels.try_reserve_exact(kernel_values).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate audio resample kernel: {error}"))
    })?;

    let orig_freq_f32 = orig_freq as f32;
    let new_freq_f32 = new_freq as f32;
    let scale = base_freq / orig_freq_f32;
    for phase in 0..new_freq {
        for kernel_index in 0..kernel_len {
            let idx = (kernel_index as f32 - width as f32) / orig_freq_f32;
            let mut t = (idx - phase as f32 / new_freq_f32) * base_freq;
            t = t.clamp(-LOWPASS_FILTER_WIDTH, LOWPASS_FILTER_WIDTH);
            let window = (t * PI_F32 / LOWPASS_FILTER_WIDTH / 2.0).cos().powi(2);
            let radians = t * PI_F32;
            let sinc = if radians == 0.0 {
                1.0
            } else {
                radians.sin() / radians
            };
            kernels.push(sinc * window * scale);
        }
    }

    let target_len_u128 = (samples.len() as u128 * new_freq as u128).div_ceil(orig_freq as u128);
    let target_len = usize::try_from(target_len_u128).map_err(|_| {
        TransformError::ShapeError("resampled audio length exceeds usize".to_string())
    })?;
    let mut output = Vec::new();
    output.try_reserve_exact(target_len).map_err(|error| {
        TransformError::ShapeError(format!("failed to allocate resampled audio: {error}"))
    })?;

    for block in 0..samples.len().div_ceil(orig_freq) {
        let input_start = block * orig_freq;
        for phase in 0..new_freq {
            if output.len() == target_len {
                return Ok(output);
            }
            let kernel = &kernels[phase * kernel_len..(phase + 1) * kernel_len];
            let mut value = 0.0_f32;
            for (kernel_index, &coefficient) in kernel.iter().enumerate() {
                let padded_index = input_start + kernel_index;
                if padded_index >= width {
                    let sample_index = padded_index - width;
                    if let Some(&sample) = samples.get(sample_index) {
                        value = sample.mul_add(coefficient, value);
                    }
                }
            }
            output.push(value);
        }
    }

    Ok(output)
}

fn greatest_common_divisor(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

#[cfg(test)]
mod tests {
    use llm_multimodal::TransformError;

    use super::bandlimited_resample;

    #[test]
    fn bandlimited_resample_matches_torchaudio_golden_vector() {
        let input = [0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.125, -0.25];
        // torchaudio 2.11 functional.resample(input, 8000, 16000), using
        // the default Hann-windowed sinc kernel.
        let expected = [
            0.012_859_77,
            0.347_330_45,
            0.230_903_15,
            -0.388_803_3,
            -0.476_089_15,
            0.325_961_17,
            0.724_112_45,
            -0.117_642_37,
            -0.975_495_4,
            -0.540_876_27,
            0.479_874_4,
            0.698_470_23,
            0.138_904_54,
            -0.281_137_47,
            -0.257_479_25,
            -0.095_786_646,
        ];

        let actual = bandlimited_resample(&input, 8_000, 16_000).unwrap();

        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "resampled sample {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn bandlimited_resample_uses_torchaudio_output_length_formula() {
        for (input_len, source_rate, destination_rate) in [
            (17, 16_000, 8_000),
            (8, 8_000, 11_025),
            (161, 48_000, 44_100),
        ] {
            let input = vec![0.0; input_len];
            let output = bandlimited_resample(&input, source_rate, destination_rate).unwrap();
            let expected_len =
                (input_len as u128 * destination_rate as u128).div_ceil(source_rate as u128);

            assert_eq!(
                output.len(),
                expected_len as usize,
                "unexpected output length for {source_rate} Hz to {destination_rate} Hz"
            );
        }
    }

    #[test]
    fn bandlimited_resample_handles_empty_and_passthrough_inputs() {
        assert_eq!(
            bandlimited_resample(&[], 8_000, 16_000).unwrap(),
            Vec::<f32>::new()
        );

        let input = [0.25, -0.5, 1.0];
        assert_eq!(bandlimited_resample(&input, 16_000, 16_000).unwrap(), input);
    }

    #[test]
    fn bandlimited_resample_rejects_zero_rates() {
        for (source_rate, destination_rate) in [(0, 16_000), (16_000, 0)] {
            assert!(matches!(
                bandlimited_resample(&[0.0], source_rate, destination_rate),
                Err(TransformError::ShapeError(message))
                    if message == "audio resampling rates must be positive"
            ));
        }
    }
}
