// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The dense primitive the Silero layers need: `out[j][r] = bias[r] + dot(w[r], x[j])`
//! for `rows` weight rows and `n` input vectors of length `k` (both row-major).
//!
//! Register-tiled SIMD paths: NEON on aarch64 (always present), AVX2+FMA on x86_64
//! (detected at run time). Elsewhere, or on x86_64 CPUs without AVX2, a portable
//! loop. Summation order differs from ONNX Runtime's GEMM, so results agree with it
//! to rounding (about 1e-5 on Silero's output), not bit for bit.
//!
//! Also the row and element-wise operations around Smart Turn's matrix products (GELU,
//! softmax, tanh, LayerNorm), written to vectorize: branch-free approximations instead of
//! libm calls, and eight running partial sums or maxima instead of a serial fold.

const LANES: usize = 8;

#[inline(always)]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    // Fused where the target guarantees hardware FMA; f32::mul_add without it
    // compiles to a slow libm call.
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

#[inline(always)]
fn hsum(acc: &[f32; LANES]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

#[inline(always)]
fn lanes(s: &[f32]) -> impl Iterator<Item = &[f32; LANES]> {
    s.chunks_exact(LANES).map(|c| <&[f32; LANES]>::try_from(c).unwrap())
}

/// Dot product of the first `min(w.len(), x.len())` elements.
#[inline(always)]
pub fn dot(w: &[f32], x: &[f32]) -> f32 {
    let k = w.len().min(x.len());
    let body = k - k % LANES;
    let mut acc = [0f32; LANES];
    for (wc, xc) in lanes(&w[..body]).zip(lanes(&x[..body])) {
        for l in 0..LANES {
            acc[l] = fmadd(wc[l], xc[l], acc[l]);
        }
    }
    let mut s = hsum(&acc);
    for i in body..k {
        s = fmadd(w[i], x[i], s);
    }
    s
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sum with eight running partial sums (vectorizes; order differs from a plain fold).
pub fn sum(x: &[f32]) -> f32 {
    let body = x.len() - x.len() % LANES;
    let mut acc = [0f32; LANES];
    for c in lanes(&x[..body]) {
        for l in 0..LANES {
            acc[l] += c[l];
        }
    }
    x[body..].iter().fold(hsum(&acc), |s, v| s + v)
}

/// Largest element, with eight running maxima. NaN-free input assumed.
#[inline(always)]
fn max(x: &[f32]) -> f32 {
    let body = x.len() - x.len() % LANES;
    let mut acc = [f32::NEG_INFINITY; LANES];
    for c in lanes(&x[..body]) {
        for l in 0..LANES {
            // Compare-and-select, not f32::max, so it maps onto the vector max instruction.
            acc[l] = if c[l] > acc[l] { c[l] } else { acc[l] };
        }
    }
    acc.iter()
        .chain(&x[body..])
        .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m })
}

/// e^x after Cephes `expf`: x = n ln2 + r with n rounded to nearest, a degree-5
/// polynomial for e^r, and 2^n written into the exponent bits. Inputs are clamped to
/// [-87, 88], so very negative x gives about 1.6e-38 instead of 0. Only IEEE-exact
/// operations, so every target computes the same bits.
#[inline(always)]
fn exp_approx(x: f32) -> f32 {
    // Adding 1.5 * 2^23 rounds to an integer held in the low mantissa bits.
    const SHIFT: f32 = 12_582_912.0;
    let x = x.clamp(-87.0, 88.0);
    let t = x * std::f32::consts::LOG2_E + SHIFT;
    let n = t - SHIFT;
    let ni = t.to_bits() as i32 - SHIFT.to_bits() as i32;
    // ln2 split in two; n * 0.693359375 is exact for |n| <= 127.
    let r = x - n * 0.693_359_4 - n * -2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 0.166_666_65)
        * r
        + 0.5;
    (p * r * r + r + 1.0) * f32::from_bits(((ni + 127) << 23) as u32)
}

/// erf after Abramowitz & Stegun 7.1.26: 1 - (a1 t + ... + a5 t^5) e^(-x^2) with
/// t = 1 / (1 + p |x|), sign restored. The formula is within 1.5e-7 of erf; in f32 the
/// result is within 6.2e-7 (libm's erff: 5e-8), which moves Smart Turn's score far less
/// than the 1e-4 parity tolerance. Absolute, not relative, accuracy (erf(0) comes out as
/// 6e-8): good where the result is added to 1, as in GELU, not as a general erf.
#[inline(always)]
fn erf_approx(x: f32) -> f32 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = ((((1.061_405_4 * t - 1.453_152) * t + 1.421_413_7) * t - 0.284_496_74) * t
        + 0.254_829_6)
        * t;
    (1.0 - poly * exp_approx(-ax * ax)).copysign(x)
}

/// GELU in the graph's op order, x * (0.5 * (erf(x / sqrt2) + 1)), except that the
/// division is a multiplication by 1/sqrt2 (at most one rounding apart, and much cheaper).
#[inline(always)]
fn gelu(x: f32) -> f32 {
    x * (0.5 * (erf_approx(x * std::f32::consts::FRAC_1_SQRT_2) + 1.0))
}

/// tanh(x) = 1 - 2 / (e^(2|x|) + 1), sign restored. Absolute error a few 1e-7 (relative
/// error is large only where |x| is tiny), saturating to exactly 1 for |x| > 9.
#[inline(always)]
fn tanh_approx(x: f32) -> f32 {
    (1.0 - 2.0 / (exp_approx(2.0 * x.abs()) + 1.0)).copysign(x)
}

#[inline(always)]
fn gelu_loop(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = gelu(*v));
}

#[inline(always)]
fn tanh_loop(x: &mut [f32]) {
    x.iter_mut().for_each(|v| *v = tanh_approx(*v));
}

#[inline(always)]
fn softmax_loop(x: &mut [f32]) {
    let m = max(x);
    x.iter_mut().for_each(|v| *v = exp_approx(*v - m));
    let r = 1.0 / sum(x);
    x.iter_mut().for_each(|v| *v *= r);
}

#[inline(always)]
fn layer_norm_loop(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    let n = w.len();
    for (row, o) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let mean = sum(row) / n as f32;
        o.iter_mut().zip(row).for_each(|(o, v)| *o = v - mean);
        let var = dot(o, o) / n as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for ((o, w), b) in o.iter_mut().zip(w).zip(b) {
            *o = *o * inv * w + b;
        }
    }
}

/// `x[i] = gelu(x[i])`.
pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU features were just checked.
        return unsafe { avx2::gelu(x) };
    }
    gelu_loop(x)
}

/// `x[i] = tanh(x[i])`, to a few 1e-7 absolute.
pub fn tanh_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU features were just checked.
        return unsafe { avx2::tanh(x) };
    }
    tanh_loop(x)
}

/// Softmax of one row in place: exp(x - max), scaled by 1 / sum.
pub fn softmax_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU features were just checked.
        return unsafe { avx2::softmax(x) };
    }
    softmax_loop(x)
}

/// LayerNorm over rows of `w.len()`: `out = (x - mean) / sqrt(var + eps) * w + b`.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU features were just checked.
        return unsafe { avx2::layer_norm(x, w, b, eps, out) };
    }
    layer_norm_loop(x, w, b, eps, out)
}

/// `out[j * rows + r] = bias[r] + dot(w[r * k..], x[j * k..])`.
pub fn linear(
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    k: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
) {
    assert_eq!(w.len(), rows * k);
    assert_eq!(x.len(), n * k);
    assert_eq!(out.len(), n * rows);
    if let Some(b) = bias {
        assert_eq!(b.len(), rows);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: shapes asserted above; NEON is part of the aarch64 baseline.
        unsafe { neon::linear(w, bias, rows, k, x, n, out) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: shapes asserted above; the CPU features were just checked.
            unsafe { avx2::linear(w, bias, rows, k, x, n, out) }
        } else {
            linear_portable(w, bias, rows, k, x, n, out)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    linear_portable(w, bias, rows, k, x, n, out)
}

#[allow(dead_code)] // unused on aarch64
fn linear_portable(
    w: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    k: usize,
    x: &[f32],
    n: usize,
    out: &mut [f32],
) {
    for j in 0..n {
        let xj = &x[j * k..(j + 1) * k];
        for r in 0..rows {
            out[j * rows + r] = dot(&w[r * k..(r + 1) * k], xj) + bias.map_or(0.0, |b| b[r]);
        }
    }
}

/// Scalar remainder `sum_{i >= from} w[r][i] * x[j][i]` for the tile kernels.
#[inline(always)]
fn tail(w: &[f32], x: &[f32], k: usize, r: usize, j: usize, from: usize) -> f32 {
    (from..k).fold(0.0f32, |s, i| fmadd(w[r * k + i], x[j * k + i], s))
}

/// NEON register tiles: four weight rows against four inputs (16 accumulators)
/// and four weight rows against one input (four accumulators per row).
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{dot, tail};
    use std::arch::aarch64::*;

    /// `k` must be a multiple of 4; returns s[row][col].
    #[inline(always)]
    unsafe fn tile_4x4(w: [*const f32; 4], x: [*const f32; 4], k: usize) -> [[f32; 4]; 4] {
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            let mut i = 0;
            while i < k {
                let xv = [
                    vld1q_f32(x[0].add(i)),
                    vld1q_f32(x[1].add(i)),
                    vld1q_f32(x[2].add(i)),
                    vld1q_f32(x[3].add(i)),
                ];
                for r in 0..4 {
                    let wv = vld1q_f32(w[r].add(i));
                    for c in 0..4 {
                        acc[r][c] = vfmaq_f32(acc[r][c], wv, xv[c]);
                    }
                }
                i += 4;
            }
            let mut s = [[0f32; 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    s[r][c] = vaddvq_f32(acc[r][c]);
                }
            }
            s
        }
    }

    /// `k` must be a multiple of 16.
    #[inline(always)]
    unsafe fn tile_4x1(w: [*const f32; 4], x: *const f32, k: usize) -> [f32; 4] {
        unsafe {
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            let mut i = 0;
            while i < k {
                let xv = [
                    vld1q_f32(x.add(i)),
                    vld1q_f32(x.add(i + 4)),
                    vld1q_f32(x.add(i + 8)),
                    vld1q_f32(x.add(i + 12)),
                ];
                for r in 0..4 {
                    for u in 0..4 {
                        acc[r][u] = vfmaq_f32(acc[r][u], vld1q_f32(w[r].add(i + 4 * u)), xv[u]);
                    }
                }
                i += 16;
            }
            let mut s = [0f32; 4];
            for r in 0..4 {
                s[r] = vaddvq_f32(vaddq_f32(
                    vaddq_f32(acc[r][0], acc[r][1]),
                    vaddq_f32(acc[r][2], acc[r][3]),
                ));
            }
            s
        }
    }

    /// Caller guarantees the shapes checked in `super::linear`.
    pub unsafe fn linear(
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        k: usize,
        x: &[f32],
        n: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let b = |r: usize| bias.map_or(0.0, |b| b[r]);
            let (k4, k16) = (k - k % 4, k - k % 16);
            let (wp, xp) = (w.as_ptr(), x.as_ptr());
            let (rows4, n4) = (rows / 4 * 4, n / 4 * 4);
            for r in (0..rows4).step_by(4) {
                let wr = [
                    wp.add(r * k),
                    wp.add((r + 1) * k),
                    wp.add((r + 2) * k),
                    wp.add((r + 3) * k),
                ];
                for j in (0..n4).step_by(4) {
                    let xj = [
                        xp.add(j * k),
                        xp.add((j + 1) * k),
                        xp.add((j + 2) * k),
                        xp.add((j + 3) * k),
                    ];
                    let s = tile_4x4(wr, xj, k4);
                    for q in 0..4 {
                        for c in 0..4 {
                            out[(j + c) * rows + r + q] =
                                s[q][c] + tail(w, x, k, r + q, j + c, k4) + b(r + q);
                        }
                    }
                }
                for j in n4..n {
                    let s = tile_4x1(wr, xp.add(j * k), k16);
                    for q in 0..4 {
                        out[j * rows + r + q] = s[q] + tail(w, x, k, r + q, j, k16) + b(r + q);
                    }
                }
            }
            for r in rows4..rows {
                for j in 0..n {
                    out[j * rows + r] = dot(&w[r * k..(r + 1) * k], &x[j * k..(j + 1) * k]) + b(r);
                }
            }
        }
    }
}

/// AVX2+FMA register tiles: two weight rows against four inputs (8 accumulators)
/// and four weight rows against one input (two accumulators per row), which fit
/// the 16 ymm registers with room for the loads.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{dot, tail};
    use std::arch::x86_64::*;

    // The element-wise loops recompiled for 256-bit vectors. No fused multiply-add is
    // formed (Rust never contracts), so these give the same bits as the portable loops.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gelu(x: &mut [f32]) {
        super::gelu_loop(x)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tanh(x: &mut [f32]) {
        super::tanh_loop(x)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax(x: &mut [f32]) {
        super::softmax_loop(x)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) {
        super::layer_norm_loop(x, w, b, eps, out)
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        unsafe {
            let mut a = [0f32; 8];
            _mm256_storeu_ps(a.as_mut_ptr(), v);
            ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
        }
    }

    /// `k` must be a multiple of 8; returns s[row][col].
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile_2x4(w: [*const f32; 2], x: [*const f32; 4], k: usize) -> [[f32; 4]; 2] {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 4]; 2];
            let mut i = 0;
            while i < k {
                let xv = [
                    _mm256_loadu_ps(x[0].add(i)),
                    _mm256_loadu_ps(x[1].add(i)),
                    _mm256_loadu_ps(x[2].add(i)),
                    _mm256_loadu_ps(x[3].add(i)),
                ];
                for r in 0..2 {
                    let wv = _mm256_loadu_ps(w[r].add(i));
                    for c in 0..4 {
                        acc[r][c] = _mm256_fmadd_ps(wv, xv[c], acc[r][c]);
                    }
                }
                i += 8;
            }
            let mut s = [[0f32; 4]; 2];
            for r in 0..2 {
                for c in 0..4 {
                    s[r][c] = hsum(acc[r][c]);
                }
            }
            s
        }
    }

    /// `k` must be a multiple of 16.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile_4x1(w: [*const f32; 4], x: *const f32, k: usize) -> [f32; 4] {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            let mut i = 0;
            while i < k {
                let xv = [_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(x.add(i + 8))];
                for r in 0..4 {
                    for u in 0..2 {
                        acc[r][u] =
                            _mm256_fmadd_ps(_mm256_loadu_ps(w[r].add(i + 8 * u)), xv[u], acc[r][u]);
                    }
                }
                i += 16;
            }
            let mut s = [0f32; 4];
            for r in 0..4 {
                s[r] = hsum(_mm256_add_ps(acc[r][0], acc[r][1]));
            }
            s
        }
    }

    /// Caller guarantees the shapes checked in `super::linear` and AVX2+FMA support.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn linear(
        w: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        k: usize,
        x: &[f32],
        n: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let b = |r: usize| bias.map_or(0.0, |b| b[r]);
            let (k8, k16) = (k - k % 8, k - k % 16);
            let (wp, xp) = (w.as_ptr(), x.as_ptr());
            let n4 = n / 4 * 4;
            // Groups of four inputs: 2-row tiles.
            let rows2 = rows / 2 * 2;
            for r in (0..rows2).step_by(2) {
                let wr = [wp.add(r * k), wp.add((r + 1) * k)];
                for j in (0..n4).step_by(4) {
                    let xj = [
                        xp.add(j * k),
                        xp.add((j + 1) * k),
                        xp.add((j + 2) * k),
                        xp.add((j + 3) * k),
                    ];
                    let s = tile_2x4(wr, xj, k8);
                    for q in 0..2 {
                        for c in 0..4 {
                            out[(j + c) * rows + r + q] =
                                s[q][c] + tail(w, x, k, r + q, j + c, k8) + b(r + q);
                        }
                    }
                }
            }
            for r in rows2..rows {
                for j in 0..n4 {
                    out[j * rows + r] = dot(&w[r * k..(r + 1) * k], &x[j * k..(j + 1) * k]) + b(r);
                }
            }
            // Leftover inputs (all of them when n < 4): 4-row tiles.
            let rows4 = rows / 4 * 4;
            for j in n4..n {
                for r in (0..rows4).step_by(4) {
                    let wr = [
                        wp.add(r * k),
                        wp.add((r + 1) * k),
                        wp.add((r + 2) * k),
                        wp.add((r + 3) * k),
                    ];
                    let s = tile_4x1(wr, xp.add(j * k), k16);
                    for q in 0..4 {
                        out[j * rows + r + q] = s[q] + tail(w, x, k, r + q, j, k16) + b(r + q);
                    }
                }
                for r in rows4..rows {
                    out[j * rows + r] = dot(&w[r * k..(r + 1) * k], &x[j * k..(j + 1) * k]) + b(r);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(w: &[f32], bias: &[f32], rows: usize, k: usize, x: &[f32], n: usize) -> Vec<f64> {
        let mut out = vec![0f64; n * rows];
        for j in 0..n {
            for r in 0..rows {
                let s: f64 = (0..k).map(|i| w[r * k + i] as f64 * x[j * k + i] as f64).sum();
                out[j * rows + r] = s + bias[r] as f64;
            }
        }
        out
    }

    #[test]
    fn linear_matches_f64_reference_on_awkward_shapes() {
        // Odd row counts, k not a multiple of 4/8/16, n below and above the tile width.
        for &(rows, k, n) in &[
            (1, 1, 1),
            (3, 5, 2),
            (7, 19, 5),
            (258, 256, 4),
            (128, 387, 4),
            (64, 384, 2),
            (130, 33, 9),
            (512, 128, 1),
        ] {
            let w: Vec<f32> =
                (0..rows * k).map(|i| ((i * 7919 % 1009) as f32 / 1009.0) - 0.5).collect();
            let x: Vec<f32> =
                (0..n * k).map(|i| ((i * 104729 % 997) as f32 / 997.0) - 0.5).collect();
            let bias: Vec<f32> = (0..rows).map(|r| r as f32 * 0.01).collect();
            let mut out = vec![0f32; n * rows];
            linear(&w, Some(&bias), rows, k, &x, n, &mut out);
            let want = reference(&w, &bias, rows, k, &x, n);
            for (i, (&g, &e)) in out.iter().zip(&want).enumerate() {
                assert!(
                    (g as f64 - e).abs() <= 1e-4 * (1.0 + e.abs()),
                    "rows {rows} k {k} n {n} at {i}: {g} vs {e}"
                );
            }
            let mut portable = vec![0f32; n * rows];
            linear_portable(&w, Some(&bias), rows, k, &x, n, &mut portable);
            for (&g, &p) in out.iter().zip(&portable) {
                assert!((g - p).abs() <= 1e-4 * (1.0 + p.abs()));
            }
        }
    }

    #[test]
    fn exp_erf_tanh_approximations_track_libm() {
        let (mut exp_rel, mut erf_abs, mut tanh_abs) = (0f64, 0f64, 0f64);
        for i in 0..=2_000_000 {
            let x = -87.0 + 175.0 * i as f32 / 2_000_000.0;
            let want = (x as f64).exp();
            exp_rel = exp_rel.max(((exp_approx(x) as f64 - want) / want).abs());
            let x = -6.0 + 12.0 * i as f32 / 2_000_000.0;
            erf_abs = erf_abs.max((erf_approx(x) as f64 - libm::erf(x as f64)).abs());
            let x = -12.0 + 24.0 * i as f32 / 2_000_000.0;
            tanh_abs = tanh_abs.max((tanh_approx(x) as f64 - (x as f64).tanh()).abs());
        }
        assert!(exp_rel <= 3e-7, "exp relative error {exp_rel:e}");
        assert!(erf_abs <= 1e-6, "erf absolute error {erf_abs:e}");
        assert!(tanh_abs <= 1e-6, "tanh absolute error {tanh_abs:e}");
        assert_eq!(exp_approx(-1e4), exp_approx(-87.0));
        assert_eq!(tanh_approx(1e4), 1.0);
        assert_eq!(tanh_approx(0.0), 0.0);
    }

    #[test]
    fn softmax_and_layer_norm_match_f64() {
        let x: Vec<f32> =
            (0..2 * 389).map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 9.0).collect();
        for row in x.chunks_exact(389) {
            let mut s = row.to_vec();
            softmax_inplace(&mut s);
            let m = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
            let z: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
            for (&g, &v) in s.iter().zip(row) {
                assert!((g as f64 - (v as f64 - m).exp() / z).abs() <= 1e-7);
            }
        }
        let w: Vec<f32> = (0..389).map(|i| 0.5 + i as f32 * 0.003).collect();
        let b: Vec<f32> = (0..389).map(|i| i as f32 * -0.001).collect();
        let mut out = vec![0f32; x.len()];
        layer_norm(&x, &w, &b, 1e-5, &mut out);
        for (row, o) in x.chunks_exact(389).zip(out.chunks_exact(389)) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / 389.0;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / 389.0;
            for i in 0..389 {
                let want = (row[i] as f64 - mean) / (var + 1e-5).sqrt() * w[i] as f64 + b[i] as f64;
                assert!(
                    (o[i] as f64 - want).abs() <= 1e-5,
                    "{i}: {} vs {want}",
                    o[i]
                );
            }
        }
    }

    #[test]
    fn element_wise_dispatch_matches_the_portable_loops_bitwise() {
        let x: Vec<f32> = (0..1003).map(|i| (i as f32 - 501.0) * 0.013).collect();
        let (mut a, mut b) = (x.clone(), x.clone());
        gelu_inplace(&mut a);
        gelu_loop(&mut b);
        assert_eq!(a, b);
        let (mut a, mut b) = (x.clone(), x.clone());
        tanh_inplace(&mut a);
        tanh_loop(&mut b);
        assert_eq!(a, b);
        let (mut a, mut b) = (x.clone(), x.clone());
        softmax_inplace(&mut a);
        softmax_loop(&mut b);
        assert_eq!(a, b);
        let (w, bias): (Vec<f32>, Vec<f32>) =
            (0..17).map(|i| (1.0 + i as f32 * 0.1, i as f32 * 0.01)).unzip();
        let (mut a, mut b) = (vec![0f32; 17 * 59], vec![0f32; 17 * 59]);
        layer_norm(&x[..17 * 59], &w, &bias, 1e-5, &mut a);
        layer_norm_loop(&x[..17 * 59], &w, &bias, 1e-5, &mut b);
        assert_eq!(a, b);
    }
}
