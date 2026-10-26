// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Pillow-exact LANCZOS resize over borrowed interleaved RGB bytes.
//!
//! Ported from `llm-multimodal`'s `vision::transforms`, whose PIL resampling
//! helpers are private there and cannot be imported.
//!
//! MuseSpark1x image/video processors resize via `PIL.Image.resize(size,
//! LANCZOS)` on the uint8 image. The SIMD `fast_image_resize` path is the same
//! filter *family* but diverges bit-wise on non-integer ratios (support
//! scaling + fixed-point details), which the vision encoder amplifies into a
//! large embedding shift. This routine replicates Pillow's `Resample.c`
//! algorithm exactly, validated against Pillow.

use image::RgbImage;
use llm_multimodal::TransformError;

const PIL_PRECISION_BITS: i64 = 32 - 8 - 2;
const PIL_LANCZOS3_SUPPORT: f64 = 3.0;

#[inline]
fn pil_lanczos3(x: f64) -> f64 {
    // Pillow's LANCZOS: sinc(x) * sinc(x / a) with a = 3.
    const A: f64 = 3.0;
    let x = x.abs();
    if x >= A {
        return 0.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    let px = std::f64::consts::PI * x;
    let ax = px / A;
    (px.sin() / px) * (ax.sin() / ax)
}

/// Pillow `precompute_coeffs` for one axis: integer (fixed-point) kernels plus
/// per-output bounds `(start, count)`.
fn pil_precompute_coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<Vec<i64>>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale >= 1.0 { scale } else { 1.0 };
    let support = PIL_LANCZOS3_SUPPORT * filterscale;
    let inv = 1.0 / filterscale;
    let coeff_scale = (1_i64 << PIL_PRECISION_BITS) as f64;

    let mut bounds = Vec::with_capacity(out_size);
    let mut kernels = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xmin = xmin as usize;
        let xmax = (xmax as usize).saturating_sub(xmin);

        let mut w = vec![0.0_f64; xmax];
        let mut tot = 0.0;
        for (x, wx) in w.iter_mut().enumerate() {
            let v = pil_lanczos3(((x + xmin) as f64 - center + 0.5) * inv);
            *wx = v;
            tot += v;
        }
        if tot != 0.0 {
            for wx in &mut w {
                *wx /= tot;
            }
        }
        // Pillow normalize_coeffs_8bpc: round half away from zero into fixed point.
        let k: Vec<i64> = w
            .iter()
            .map(|&c| {
                if c < 0.0 {
                    (-0.5 + c * coeff_scale) as i64
                } else {
                    (0.5 + c * coeff_scale) as i64
                }
            })
            .collect();
        bounds.push((xmin, xmax));
        kernels.push(k);
    }
    (bounds, kernels)
}

#[inline]
fn pil_clip8(v: i64) -> u8 {
    let v = v >> PIL_PRECISION_BITS;
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

const PARALLEL_MIN_BYTES: usize = 1 << 19;
const MAX_TASKS_PER_OPERATION: usize = 8;

/// Number of threads to split an elementwise or row-banded preprocessing pass
/// across. Each output row/element is independent, so banding work over threads
/// yields BIT-IDENTICAL output: no shared accumulation and no inner-loop order
/// changes. Small images run serial to avoid thread-spawn overhead.
fn par_threads(out_bytes: usize, out_rows: usize) -> usize {
    const MIN_ROWS_PER_TASK: usize = 32;
    if out_bytes < PARALLEL_MIN_BYTES || out_rows < 2 * MIN_ROWS_PER_TASK {
        return 1;
    }
    (out_rows / MIN_ROWS_PER_TASK)
        .min(std::thread::available_parallelism().map_or(1, usize::from))
        .clamp(1, MAX_TASKS_PER_OPERATION)
}

#[expect(
    clippy::too_many_arguments,
    reason = "RGB row-band resampler: precomputed coeffs + dims + output band"
)]
fn pil_h_band_rgb(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    in_w: usize,
    out_w: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = out_w * 3;
    for (i, output_row) in out_band.chunks_mut(row_out).enumerate() {
        let y = oy0 + i;
        let row = &src[y * in_w * 3..(y + 1) * in_w * 3];
        for output_x in 0..out_w {
            let (source_x, source_columns) = bounds[output_x];
            let kernel = &kernels[output_x];
            let mut red = half;
            let mut green = half;
            let mut blue = half;
            let source_start = source_x * 3;
            let source_end = (source_x + source_columns) * 3;
            for (pixel, &coefficient) in
                row[source_start..source_end].as_chunks::<3>().0.iter().zip(kernel)
            {
                red += pixel[0] as i64 * coefficient;
                green += pixel[1] as i64 * coefficient;
                blue += pixel[2] as i64 * coefficient;
            }
            let output = output_x * 3;
            output_row[output] = pil_clip8(red);
            output_row[output + 1] = pil_clip8(green);
            output_row[output + 2] = pil_clip8(blue);
        }
    }
}

/// Resample interleaved RGB u8 data along the width axis.
/// `src` is `rows * in_w * 3`; returns `rows * out_w * 3`.
fn pil_resample_horizontal_rgb(src: &[u8], rows: usize, in_w: usize, out_w: usize) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_w, out_w);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = out_w * 3;
    let mut out = vec![0_u8; rows * row_out];
    let nthreads = par_threads(out.len(), rows);
    if nthreads <= 1 {
        pil_h_band_rgb(src, &bounds, &kernels, half, in_w, out_w, 0, &mut out);
    } else {
        let chunk_rows = rows.div_ceil(nthreads);
        std::thread::scope(|scope| {
            let (bounds, kernels) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut output_y = 0;
            while output_y < rows {
                let band_rows = chunk_rows.min(rows - output_y);
                let (band, tail) = rest.split_at_mut(band_rows * row_out);
                rest = tail;
                let start = output_y;
                scope.spawn(move || {
                    pil_h_band_rgb(src, bounds, kernels, half, in_w, out_w, start, band);
                });
                output_y += band_rows;
            }
        });
    }
    out
}

fn pil_v_band_rgb(
    src: &[u8],
    bounds: &[(usize, usize)],
    kernels: &[Vec<i64>],
    half: i64,
    width: usize,
    oy0: usize,
    out_band: &mut [u8],
) {
    let row_out = width * 3;
    for (i, output_row) in out_band.chunks_mut(row_out).enumerate() {
        let output_y = oy0 + i;
        let (source_y, source_rows) = bounds[output_y];
        let kernel = &kernels[output_y];
        let blocked_width = width / 4 * 4;
        for x in (0..blocked_width).step_by(4) {
            let mut sums = [[half; 3]; 4];
            for (y, &coefficient) in kernel.iter().take(source_rows).enumerate() {
                let source = ((source_y + y) * width + x) * 3;
                for (pixel, sums) in sums.iter_mut().enumerate() {
                    let input = source + pixel * 3;
                    sums[0] += src[input] as i64 * coefficient;
                    sums[1] += src[input + 1] as i64 * coefficient;
                    sums[2] += src[input + 2] as i64 * coefficient;
                }
            }
            let output = x * 3;
            for (pixel, sums) in sums.iter().enumerate() {
                let target = output + pixel * 3;
                output_row[target] = pil_clip8(sums[0]);
                output_row[target + 1] = pil_clip8(sums[1]);
                output_row[target + 2] = pil_clip8(sums[2]);
            }
        }
        for x in blocked_width..width {
            let mut red = half;
            let mut green = half;
            let mut blue = half;
            for (y, &coefficient) in kernel.iter().take(source_rows).enumerate() {
                let source = ((source_y + y) * width + x) * 3;
                red += src[source] as i64 * coefficient;
                green += src[source + 1] as i64 * coefficient;
                blue += src[source + 2] as i64 * coefficient;
            }
            let output = x * 3;
            output_row[output] = pil_clip8(red);
            output_row[output + 1] = pil_clip8(green);
            output_row[output + 2] = pil_clip8(blue);
        }
    }
}

/// Resample interleaved RGB u8 data along the height axis.
/// `src` is `in_h * width * 3`; returns `out_h * width * 3`.
fn pil_resample_vertical_rgb(src: &[u8], in_h: usize, width: usize, out_h: usize) -> Vec<u8> {
    let (bounds, kernels) = pil_precompute_coeffs(in_h, out_h);
    let half = 1_i64 << (PIL_PRECISION_BITS - 1);
    let row_out = width * 3;
    let mut out = vec![0_u8; out_h * row_out];
    let nthreads = par_threads(out.len(), out_h);
    if nthreads <= 1 {
        pil_v_band_rgb(src, &bounds, &kernels, half, width, 0, &mut out);
    } else {
        let chunk_rows = out_h.div_ceil(nthreads);
        std::thread::scope(|scope| {
            let (bounds, kernels) = (&bounds, &kernels);
            let mut rest = out.as_mut_slice();
            let mut output_y = 0;
            while output_y < out_h {
                let rows = chunk_rows.min(out_h - output_y);
                let (band, tail) = rest.split_at_mut(rows * row_out);
                rest = tail;
                let start = output_y;
                scope.spawn(move || {
                    pil_v_band_rgb(src, bounds, kernels, half, width, start, band);
                });
                output_y += rows;
            }
        });
    }
    out
}

/// PIL-exact LANCZOS resize over borrowed interleaved RGB bytes.
///
/// Matches `PIL.Image.resize(.., LANCZOS)` byte-for-byte; used by both the
/// image path and the raw-RGB video frame path (`preprocess_video_rgb`).
pub(super) fn resize_lanczos_pil_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    out_w: u32,
    out_h: u32,
) -> Result<RgbImage, TransformError> {
    let (in_w, in_h) = (width as usize, height as usize);
    let expected = in_w.saturating_mul(in_h).saturating_mul(3);
    if data.len() != expected {
        return Err(TransformError::ShapeError(format!(
            "PIL LANCZOS RGB source has {} bytes, expected {expected} for {width}x{height}",
            data.len()
        )));
    }
    let output = resize_lanczos_pil_bytes(data, width, height, out_w, out_h);
    RgbImage::from_raw(out_w, out_h, output).ok_or_else(|| {
        TransformError::ShapeError(format!(
            "failed to build PIL LANCZOS RGB image for {out_w}x{out_h}"
        ))
    })
}

fn resize_lanczos_pil_bytes(data: &[u8], in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> Vec<u8> {
    let (in_w, in_h, out_w, out_h) = (in_w as usize, in_h as usize, out_w as usize, out_h as usize);
    if in_w == out_w && in_h == out_h {
        data.to_vec()
    } else if in_w == out_w {
        pil_resample_vertical_rgb(data, in_h, in_w, out_h)
    } else {
        let horiz = pil_resample_horizontal_rgb(data, in_h, in_w, out_w);
        if in_h == out_h {
            horiz
        } else {
            pil_resample_vertical_rgb(&horiz, in_h, out_w, out_h)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pillow LANCZOS golden: 5x4 upscaled to 11x7 (non-integer ratios, both
    /// axes). Expected bytes were produced by Pillow 12.3:
    /// `Image.fromarray(src).resize((11, 7), Image.Resampling.LANCZOS)`.
    #[test]
    fn resize_lanczos_pil_matches_pillow() {
        #[rustfmt::skip]
        let src: [u8; 60] = [
            102, 220, 225, 95, 179, 61, 234, 203, 92, 3, 98, 243,
            14, 149, 245, 46, 106, 244, 99, 187, 71, 212, 153, 199,
            188, 174, 65, 153, 20, 44, 203, 152, 102, 214, 240, 39,
            121, 24, 34, 114, 210, 65, 239, 39, 214, 244, 151, 25,
            74, 145, 222, 14, 202, 85, 145, 117, 87, 184, 189, 221,
        ];
        #[rustfmt::skip]
        let expected: [u8; 231] = [
            122, 244, 243, 106, 226, 197, 83, 195, 125, 111, 180, 45,
            204, 202, 25, 233, 204, 74, 137, 150, 160, 0, 94, 251,
            0, 107, 255, 0, 150, 255, 14, 179, 255, 78, 185, 255,
            69, 180, 216, 60, 174, 130, 99, 179, 61, 191, 199, 79,
            233, 201, 137, 179, 168, 175, 79, 123, 190, 50, 105, 185,
            42, 109, 177, 52, 115, 168, 42, 100, 255, 45, 122, 219,
            59, 158, 125, 104, 187, 74, 172, 177, 137, 216, 164, 199,
            218, 179, 170, 187, 175, 89, 159, 114, 50, 143, 47, 48,
            136, 12, 51, 103, 90, 200, 119, 136, 162, 151, 206, 79,
            182, 222, 28, 189, 120, 76, 179, 56, 124, 168, 131, 94,
            170, 217, 40, 190, 150, 50, 217, 27, 87, 230, 0, 112, 209,
            136, 98, 216, 175, 87, 222, 231, 69, 203, 224, 45, 157,
            105, 32, 111, 31, 29, 83, 100, 25, 100, 199, 55, 163,
            169, 134, 221, 79, 205, 253, 23, 244, 251, 152, 21, 233,
            161, 58, 183, 173, 128, 109, 176, 166, 52, 158, 121, 39,
            140, 56, 61, 138, 31, 115, 145, 69, 170, 148, 148, 199,
            148, 216, 212, 148, 255, 255, 155, 0, 221, 143, 56, 137,
            126, 173, 35, 140, 249, 0, 201, 193, 5, 227, 97, 64, 173,
            54, 139, 108, 78, 177, 128, 142, 177, 187, 200, 172, 225, 234,
        ];
        let img = RgbImage::from_raw(5, 4, src.to_vec()).unwrap();
        let via_bytes = resize_lanczos_pil_rgb(img.as_raw(), 5, 4, 11, 7).unwrap();
        assert_eq!(via_bytes.into_raw(), expected.to_vec());
    }
}
