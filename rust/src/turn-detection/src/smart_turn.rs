// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Smart Turn v3.2, fp32 graph (`smart-turn-v3.2-gpu.onnx`), as a hand-written forward pass,
//! plus the Whisper log-mel front end (`WhisperFeatureExtractor(chunk_length=8)`).
//!
//! Network, read from the ONNX graph:
//!
//! ```text
//! [80, 800] log-mel -> Conv1d(80->384, k3, p1) + GELU -> Conv1d(384->384, k3, s2, p1) + GELU
//!   -> [400, 384] + positions -> 4 x pre-LN Whisper encoder layer (6 heads of 64, FFN 1536, erf GELU)
//!   -> LayerNorm -> attention pooling: softmax_t(Linear(256->1)(tanh(Linear(384->256)(x)))) . x
//!   -> Linear(384->256) -> LayerNorm -> GELU -> Linear(256->64) -> GELU -> Linear(64->1) -> sigmoid
//! ```
//!
//! The int8 graph (`smart-turn-v3.2-cpu.onnx`) is not reproduced: its uint8 activation
//! quantization makes the output depend on rounding details of the runtime.

use crate::kernels;
use crate::onnx::{Tensor, load_graph};
use crate::{Result, TurnDetectionError, pins, read_model};
use std::collections::HashMap;
use std::path::Path;

pub const SAMPLES: usize = 128_000; // 8 s at 16 kHz
pub const N_MELS: usize = 80;
pub const FRAMES: usize = 800;
const D: usize = 384;
const T: usize = 400;
const HEADS: usize = 6;
const HD: usize = 64;
const FF: usize = 1536;
const LN_EPS: f32 = 1e-5;

struct Layer {
    ln1: (Vec<f32>, Vec<f32>),
    wq: Vec<f32>, // [in][out], as stored by the ONNX MatMul; wq, bq and wk pre-scaled by 64^-0.25
    bq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    bv: Vec<f32>,
    wo: Vec<f32>,
    bo: Vec<f32>,
    ln2: (Vec<f32>, Vec<f32>),
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

pub struct SmartTurnModel {
    conv1_w: Vec<f32>, // [384][80 * 3]
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>, // [384][384 * 3]
    conv2_b: Vec<f32>,
    pos: Vec<f32>, // [400][384]
    layers: Vec<Layer>,
    ln_f: (Vec<f32>, Vec<f32>),
    pool0_w: Vec<f32>, // [384][256]
    pool0_b: Vec<f32>,
    pool2_w: Vec<f32>, // [256][1]
    pool2_b: f32,
    cls0_w: Vec<f32>, // [256][384] (Gemm, transB)
    cls0_b: Vec<f32>,
    cls1: (Vec<f32>, Vec<f32>),
    cls4_w: Vec<f32>, // [64][256]
    cls4_b: Vec<f32>,
    cls6_w: Vec<f32>, // [1][64]
    cls6_b: f32,
}

fn take(
    t: &mut HashMap<String, Tensor>,
    name: &str,
    dims: &[usize],
) -> std::result::Result<Vec<f32>, String> {
    let tensor = t.remove(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if tensor.dims != dims {
        return Err(format!("{name}: dims {:?}, expected {dims:?}", tensor.dims));
    }
    Ok(tensor.f32()?.to_vec())
}

/// `c[m][n] (+)= a[m][k] . b[k][n]` with arbitrary strides (row stride, column stride).
#[inline]
fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    c: &mut [f32],
    rsc: usize,
    beta: f32,
) {
    // Bounds: the last element each operand touches must be in range.
    assert!(m > 0 && k > 0 && n > 0);
    assert!((m - 1) * rsa + (k - 1) * csa < a.len());
    assert!((k - 1) * rsb + (n - 1) * csb < b.len());
    assert!((m - 1) * rsc + n - 1 < c.len());
    // SAFETY: the asserts above keep every access inside the slices.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            rsa as isize,
            csa as isize,
            b.as_ptr(),
            rsb as isize,
            csb as isize,
            beta,
            c.as_mut_ptr(),
            rsc as isize,
            1,
        );
    }
}

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        for (v, bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32]) {
    kernels::layer_norm(x, w, b, LN_EPS, out)
}

impl SmartTurnModel {
    /// Load the pinned fp32 `smart-turn-v3.2-gpu.onnx`, checking its sha256 ([`pins::SMART_TURN_SHA256`]).
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_onnx_bytes(&read_model(path, Some(pins::SMART_TURN_SHA256))?)
    }

    fn from_onnx_bytes(model: &[u8]) -> Result<Self> {
        Self::parse(model).map_err(|message| TurnDetectionError::Model { message })
    }

    fn parse(model: &[u8]) -> std::result::Result<Self, String> {
        let (mut t, nodes) = load_graph(model)?;
        // The projection weights are anonymous MatMul initializers; take them in graph order.
        let anon: Vec<String> = nodes
            .iter()
            .filter(|n| {
                n.op == "MatMul"
                    && n.inputs.len() == 2
                    && t.contains_key(&n.inputs[1])
                    && n.inputs[1].starts_with("val_")
            })
            .map(|n| n.inputs[1].clone())
            .collect();
        if anon.len() != 4 * 6 + 2 {
            return Err(format!(
                "expected 26 anonymous MatMul weights, found {}",
                anon.len()
            ));
        }
        let mut anon = anon.into_iter();
        let mut next =
            |t: &mut HashMap<String, Tensor>, dims: &[usize]| take(t, &anon.next().unwrap(), dims);
        let p = "inner.encoder.";
        // The graph scales q (bias included) and k by 64^-0.25 after the projections; fold
        // that into the weights once here.
        let scale = (HD as f32).powf(-0.25);
        let scaled = |mut v: Vec<f32>| {
            v.iter_mut().for_each(|e| *e *= scale);
            v
        };
        let mut layers = Vec::new();
        for l in 0..4 {
            let q = format!("{p}layers.{l}.");
            let ln1 = (
                take(&mut t, &format!("{q}self_attn_layer_norm.weight"), &[D])?,
                take(&mut t, &format!("{q}self_attn_layer_norm.bias"), &[D])?,
            );
            let wq = scaled(next(&mut t, &[D, D])?);
            let wk = scaled(next(&mut t, &[D, D])?);
            let wv = next(&mut t, &[D, D])?;
            let wo = next(&mut t, &[D, D])?;
            let w1 = next(&mut t, &[D, FF])?;
            let w2 = next(&mut t, &[FF, D])?;
            layers.push(Layer {
                ln1,
                wq,
                bq: scaled(take(&mut t, &format!("{q}self_attn.q_proj.bias"), &[D])?),
                wk,
                wv,
                bv: take(&mut t, &format!("{q}self_attn.v_proj.bias"), &[D])?,
                wo,
                bo: take(&mut t, &format!("{q}self_attn.out_proj.bias"), &[D])?,
                ln2: (
                    take(&mut t, &format!("{q}final_layer_norm.weight"), &[D])?,
                    take(&mut t, &format!("{q}final_layer_norm.bias"), &[D])?,
                ),
                w1,
                b1: take(&mut t, &format!("{q}fc1.bias"), &[FF])?,
                w2,
                b2: take(&mut t, &format!("{q}fc2.bias"), &[D])?,
            });
        }
        let pool0_w = next(&mut t, &[D, 256])?;
        let pool2_w = next(&mut t, &[256, 1])?;
        Ok(Self {
            conv1_w: take(&mut t, &format!("{p}conv1.weight"), &[D, N_MELS, 3])?,
            conv1_b: take(&mut t, &format!("{p}conv1.bias"), &[D])?,
            conv2_w: take(&mut t, &format!("{p}conv2.weight"), &[D, D, 3])?,
            conv2_b: take(&mut t, &format!("{p}conv2.bias"), &[D])?,
            pos: take(&mut t, &format!("{p}embed_positions.weight"), &[T, D])?,
            layers,
            ln_f: (
                take(&mut t, &format!("{p}layer_norm.weight"), &[D])?,
                take(&mut t, &format!("{p}layer_norm.bias"), &[D])?,
            ),
            pool0_w,
            pool0_b: take(&mut t, "inner.pool_attention.0.bias", &[256])?,
            pool2_w,
            pool2_b: take(&mut t, "inner.pool_attention.2.bias", &[1])?[0],
            cls0_w: take(&mut t, "inner.classifier.0.weight", &[256, D])?,
            cls0_b: take(&mut t, "inner.classifier.0.bias", &[256])?,
            cls1: (
                take(&mut t, "inner.classifier.1.weight", &[256])?,
                take(&mut t, "inner.classifier.1.bias", &[256])?,
            ),
            cls4_w: take(&mut t, "inner.classifier.4.weight", &[64, 256])?,
            cls4_b: take(&mut t, "inner.classifier.4.bias", &[64])?,
            cls6_w: take(&mut t, "inner.classifier.6.weight", &[1, 64])?,
            cls6_b: take(&mut t, "inner.classifier.6.bias", &[1])?[0],
        })
    }

    /// Probability that the turn is complete, from `[80][800]` log-mel features (mel-major).
    pub fn score(&self, feat: &[f32]) -> f32 {
        assert_eq!(feat.len(), N_MELS * FRAMES);
        // conv1 as im2col + GEMM: cols [800][80*3] . W1^T -> [800][384] (token-major).
        let k1 = N_MELS * 3;
        let mut cols = vec![0f32; FRAMES * k1];
        for t in 0..FRAMES {
            for c in 0..N_MELS {
                for kk in 0..3 {
                    let p = t as isize + kk as isize - 1;
                    if (0..FRAMES as isize).contains(&p) {
                        cols[t * k1 + c * 3 + kk] = feat[c * FRAMES + p as usize];
                    }
                }
            }
        }
        let mut h1 = vec![0f32; FRAMES * D];
        gemm(
            FRAMES,
            k1,
            D,
            &cols,
            k1,
            1,
            &self.conv1_w,
            1,
            k1,
            &mut h1,
            D,
            0.0,
        );
        add_bias(&mut h1, &self.conv1_b);
        kernels::gelu_inplace(&mut h1);
        // conv2, stride 2: cols [400][384*3].
        let k2 = D * 3;
        let mut cols = vec![0f32; T * k2];
        for t in 0..T {
            for kk in 0..3 {
                let p = (2 * t + kk) as isize - 1;
                if (0..FRAMES as isize).contains(&p) {
                    let src = &h1[p as usize * D..(p as usize + 1) * D];
                    for c in 0..D {
                        cols[t * k2 + c * 3 + kk] = src[c];
                    }
                }
            }
        }
        let mut x = vec![0f32; T * D];
        gemm(T, k2, D, &cols, k2, 1, &self.conv2_w, 1, k2, &mut x, D, 0.0);
        add_bias(&mut x, &self.conv2_b);
        kernels::gelu_inplace(&mut x);
        x.iter_mut().zip(&self.pos).for_each(|(v, p)| *v += p);

        let mut h = vec![0f32; T * D];
        let (mut q, mut k, mut v, mut att) = (
            vec![0f32; T * D],
            vec![0f32; T * D],
            vec![0f32; T * D],
            vec![0f32; T * D],
        );
        let mut s = vec![0f32; T * T];
        let mut f = vec![0f32; T * FF];
        for l in &self.layers {
            layer_norm(&x, &l.ln1.0, &l.ln1.1, &mut h);
            gemm(T, D, D, &h, D, 1, &l.wq, D, 1, &mut q, D, 0.0);
            gemm(T, D, D, &h, D, 1, &l.wk, D, 1, &mut k, D, 0.0);
            gemm(T, D, D, &h, D, 1, &l.wv, D, 1, &mut v, D, 0.0);
            add_bias(&mut q, &l.bq);
            add_bias(&mut v, &l.bv);
            for hd in 0..HEADS {
                let o = hd * HD;
                // s = q_h . k_h^T: [400][64] . [64][400]
                gemm(T, HD, T, &q[o..], D, 1, &k[o..], 1, D, &mut s, T, 0.0);
                for row in s.chunks_exact_mut(T) {
                    kernels::softmax_inplace(row);
                }
                gemm(T, T, HD, &s, T, 1, &v[o..], D, 1, &mut att[o..], D, 0.0);
            }
            // x += att . Wo + bo
            gemm(T, D, D, &att, D, 1, &l.wo, D, 1, &mut h, D, 0.0);
            add_bias(&mut h, &l.bo);
            x.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
            layer_norm(&x, &l.ln2.0, &l.ln2.1, &mut h);
            gemm(T, D, FF, &h, D, 1, &l.w1, FF, 1, &mut f, FF, 0.0);
            add_bias(&mut f, &l.b1);
            kernels::gelu_inplace(&mut f);
            gemm(T, FF, D, &f, FF, 1, &l.w2, D, 1, &mut h, D, 0.0);
            add_bias(&mut h, &l.b2);
            x.iter_mut().zip(&h).for_each(|(a, b)| *a += b);
        }
        layer_norm(&x, &self.ln_f.0, &self.ln_f.1, &mut h);

        // Attention pooling over the 400 positions.
        let mut a = vec![0f32; T * 256];
        gemm(T, D, 256, &h, D, 1, &self.pool0_w, 256, 1, &mut a, 256, 0.0);
        add_bias(&mut a, &self.pool0_b);
        kernels::tanh_inplace(&mut a);
        let mut w = vec![0f32; T];
        gemm(T, 256, 1, &a, 256, 1, &self.pool2_w, 1, 1, &mut w, 1, 0.0);
        w.iter_mut().for_each(|e| *e += self.pool2_b);
        kernels::softmax_inplace(&mut w);
        let mut pooled = [0f32; D];
        for t in 0..T {
            for c in 0..D {
                pooled[c] += h[t * D + c] * w[t];
            }
        }

        // Classifier.
        let lin = |w: &[f32], b: &[f32], x: &[f32]| -> Vec<f32> {
            b.iter()
                .enumerate()
                .map(|(r, bb)| crate::kernels::dot(&w[r * x.len()..(r + 1) * x.len()], x) + bb)
                .collect()
        };
        let y = lin(&self.cls0_w, &self.cls0_b, &pooled);
        let mut yn = vec![0f32; 256];
        layer_norm(&y, &self.cls1.0, &self.cls1.1, &mut yn);
        kernels::gelu_inplace(&mut yn);
        let mut z = lin(&self.cls4_w, &self.cls4_b, &yn);
        kernels::gelu_inplace(&mut z);
        let o = crate::kernels::dot(&self.cls6_w, &z) + self.cls6_b;
        crate::kernels::sigmoid(o)
    }
}

/// The model's input window: the last 8 s of `audio`, zero-padded at the front, so
/// the speech always ends at the end of the window (vllm-omni `smart_turn_window`).
pub fn window(audio: &[f32]) -> Vec<f32> {
    let mut w = vec![0f32; SAMPLES];
    let n = audio.len().min(SAMPLES);
    w[SAMPLES - n..].copy_from_slice(&audio[audio.len() - n..]);
    w
}

/// Whisper log-mel features of exactly 8 s of 16 kHz audio, as computed by
/// `WhisperFeatureExtractor(chunk_length=8)` with `do_normalize=True`:
/// zero-mean/unit-variance waveform, centered STFT (n_fft 400, hop 160, periodic Hann,
/// reflect padding), power spectrum, Slaney mel bank (80 mels, 0-8 kHz), log10 floored at
/// max - 8, then (x + 4) / 4. Output `[80][800]`, mel-major.
pub struct LogMel {
    window: Vec<f32>,
    mel: Vec<f32>,              // [80][201]
    bands: Vec<(usize, usize)>, // per mel filter, the bin range where it is non-zero
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

const N_FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;

fn hz_to_mel(f: f64) -> f64 {
    // Slaney scale: linear below 1 kHz, log above.
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// f64 sum with four running partial sums, so the additions do not form one serial chain.
fn sum_f64(x: impl Iterator<Item = f64>) -> f64 {
    let mut acc = [0f64; 4];
    for (i, v) in x.enumerate() {
        acc[i % 4] += v;
    }
    (acc[0] + acc[2]) + (acc[1] + acc[3])
}

impl Default for LogMel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogMel {
    pub fn new() -> Self {
        // transformers.audio_utils.mel_filter_bank(201, 80, 0, 8000, 16000, norm="slaney", mel_scale="slaney"), in f64.
        let fft_freqs: Vec<f64> =
            (0..BINS).map(|i| 8000.0 * i as f64 / (BINS - 1) as f64).collect();
        let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let filter_freqs: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut mel = vec![0f32; N_MELS * BINS];
        for m in 0..N_MELS {
            let (lo, ce, hi) = (filter_freqs[m], filter_freqs[m + 1], filter_freqs[m + 2]);
            let enorm = 2.0 / (hi - lo);
            for (b, &f) in fft_freqs.iter().enumerate() {
                let down = (f - lo) / (ce - lo);
                let up = (hi - f) / (hi - ce);
                mel[m * BINS + b] = (down.min(up).max(0.0) * enorm) as f32;
            }
        }
        let bands = mel
            .chunks_exact(BINS)
            .map(|f| match f.iter().position(|&v| v != 0.0) {
                Some(lo) => (lo, BINS - f.iter().rev().position(|&v| v != 0.0).unwrap()),
                None => (0, 0),
            })
            .collect();
        let window = (0..N_FFT)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Self {
            window,
            mel,
            bands,
            fft,
        }
    }

    pub fn compute(&self, audio: &[f32]) -> Vec<f32> {
        assert_eq!(audio.len(), SAMPLES);
        let n = SAMPLES as f64;
        let mean = sum_f64(audio.iter().map(|&v| v as f64)) / n;
        let var = sum_f64(audio.iter().map(|&v| (v as f64 - mean).powi(2))) / n;
        let (mean, inv) = (mean as f32, 1.0 / ((var as f32) + 1e-7).sqrt());
        let pad = N_FFT / 2;
        let mut x = vec![0f32; SAMPLES + 2 * pad];
        for i in 0..SAMPLES {
            x[pad + i] = (audio[i] - mean) * inv;
        }
        for i in 0..pad {
            x[pad - 1 - i] = x[pad + 1 + i];
            x[pad + SAMPLES + i] = x[pad + SAMPLES - 2 - i];
        }
        let mut frame = vec![0f32; N_FFT];
        let mut spec = self.fft.make_output_vec();
        let mut power = vec![0f32; BINS];
        let mut out = vec![0f32; N_MELS * FRAMES];
        for t in 0..FRAMES {
            for i in 0..N_FFT {
                frame[i] = x[t * HOP + i] * self.window[i];
            }
            self.fft.process(&mut frame, &mut spec).unwrap();
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.re * c.re + c.im * c.im;
            }
            for (m, &(lo, hi)) in self.bands.iter().enumerate() {
                let e =
                    crate::kernels::dot(&self.mel[m * BINS + lo..m * BINS + hi], &power[lo..hi]);
                out[m * FRAMES + t] = e.max(1e-10).log10();
            }
        }
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in out.iter_mut() {
            *v = (v.max(max - 8.0) + 4.0) / 4.0;
        }
        out
    }
}
