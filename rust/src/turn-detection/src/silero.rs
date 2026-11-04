// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Silero VAD v6.2, 16 kHz path, as a hand-written forward pass.
//!
//! Same contract as vllm-omni's `SileroVADBackend`: each call scores 512 new
//! samples, prefixed by the last 64 samples of the previous call, and carries the
//! LSTM state `[h, c]` (the model's `[2, 1, 128]` state). The network, read from
//! the `If` then-branch (sr == 16000) of the pinned `silero_vad.onnx`:
//!
//! ```text
//! 576 samples -> reflect-pad right 64 -> STFT as Conv1d(1->258, k256, s128): 4 frames
//!   -> magnitude of 129 bins -> 4x [Conv1d(k3, pad1) + ReLU]: 129->128 s1, 128->64 s2,
//!   64->64 s2, 64->128 s1 (time 4 -> 4 -> 2 -> 1 -> 1)
//!   -> LSTMCell(128, 128) -> ReLU -> Conv1d(128->1, k1) -> sigmoid -> mean over time
//! ```

use crate::kernels::{linear, sigmoid};
use crate::onnx::{Tensor, load_tensors};
use crate::{Result, TurnDetectionError, pins, read_model};
use std::collections::HashMap;
use std::path::Path;

pub const FRAME: usize = 512;
pub const CONTEXT: usize = 64;
pub const HIDDEN: usize = 128;
const INPUT: usize = CONTEXT + FRAME; // 576
const PADDED: usize = INPUT + 64; // 640
const N_FFT: usize = 256;
const HOP: usize = 128;
const FRAMES: usize = (PADDED - N_FFT) / HOP + 1; // 4
const BINS: usize = N_FFT / 2 + 1; // 129

struct Conv {
    w: Vec<f32>, // [out][in][3]
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    stride: usize,
}

pub struct SileroModel {
    basis: Vec<f32>, // [258][256]
    convs: [Conv; 4],
    w_ih: Vec<f32>, // [512][128], PyTorch gate order i, f, g, o
    w_hh: Vec<f32>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    dec_w: Vec<f32>, // [128]
    dec_b: f32,
}

/// Per-stream state: the LSTM `[h, c]` and the 64-sample context.
#[derive(Clone)]
pub struct SileroState {
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
    context: [f32; CONTEXT],
}

impl Default for SileroState {
    fn default() -> Self {
        Self {
            h: [0.0; HIDDEN],
            c: [0.0; HIDDEN],
            context: [0.0; CONTEXT],
        }
    }
}

/// Reusable buffers for one caller (no allocation per call once sized).
#[derive(Default)]
pub struct SileroScratch {
    cols: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    gx: Vec<f32>,
    gh: Vec<f32>,
    hs: Vec<f32>,
}

fn take(
    t: &mut HashMap<String, Tensor>,
    name: &str,
    dims: &[usize],
) -> std::result::Result<Vec<f32>, String> {
    let key = format!("If_0_then_branch__Inline_0__{name}");
    let tensor = t.remove(&key).ok_or_else(|| format!("missing tensor {key}"))?;
    if tensor.dims != dims {
        return Err(format!("{name}: dims {:?}, expected {dims:?}", tensor.dims));
    }
    Ok(tensor.f32()?.to_vec())
}

impl SileroModel {
    /// Load the pinned `silero_vad.onnx`, checking its sha256 ([`pins::SILERO_SHA256`]).
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_onnx_bytes(&read_model(path, Some(pins::SILERO_SHA256))?)
    }

    fn from_onnx_bytes(model: &[u8]) -> Result<Self> {
        Self::parse(model).map_err(|message| TurnDetectionError::Model { message })
    }

    fn parse(model: &[u8]) -> std::result::Result<Self, String> {
        let mut t = load_tensors(model)?;
        let conv = |t: &mut HashMap<String, Tensor>,
                    i: usize,
                    cin: usize,
                    cout: usize,
                    stride: usize|
         -> std::result::Result<Conv, String> {
            Ok(Conv {
                w: take(
                    t,
                    &format!("encoder.{i}.reparam_conv.weight"),
                    &[cout, cin, 3],
                )?,
                b: take(t, &format!("encoder.{i}.reparam_conv.bias"), &[cout])?,
                cin,
                cout,
                stride,
            })
        };
        Ok(Self {
            basis: take(&mut t, "stft.forward_basis_buffer", &[2 * BINS, 1, N_FFT])?,
            convs: [
                conv(&mut t, 0, BINS, 128, 1)?,
                conv(&mut t, 1, 128, 64, 2)?,
                conv(&mut t, 2, 64, 64, 2)?,
                conv(&mut t, 3, 64, 128, 1)?,
            ],
            w_ih: take(&mut t, "decoder.rnn.weight_ih", &[4 * HIDDEN, HIDDEN])?,
            w_hh: take(&mut t, "decoder.rnn.weight_hh", &[4 * HIDDEN, HIDDEN])?,
            b_ih: take(&mut t, "decoder.rnn.bias_ih", &[4 * HIDDEN])?,
            b_hh: take(&mut t, "decoder.rnn.bias_hh", &[4 * HIDDEN])?,
            dec_w: take(&mut t, "decoder.decoder.2.weight", &[1, HIDDEN, 1])?,
            dec_b: take(&mut t, "decoder.decoder.2.bias", &[1])?[0],
        })
    }

    /// Score one frame of one stream.
    pub fn infer(
        &self,
        frame: &[f32],
        state: &mut SileroState,
        scratch: &mut SileroScratch,
    ) -> f32 {
        let mut p = [0f32];
        self.infer_batch(&[frame], std::slice::from_mut(state), scratch, &mut p);
        p[0]
    }

    /// Score one frame for each of `frames.len()` independent streams.
    fn infer_batch(
        &self,
        frames: &[&[f32]],
        states: &mut [SileroState],
        s: &mut SileroScratch,
        out: &mut [f32],
    ) {
        let n = frames.len();
        assert_eq!(states.len(), n);
        assert_eq!(out.len(), n);

        // STFT frames: [n * 4][256] windows of the padded input.
        s.cols.resize(n * FRAMES * N_FFT, 0.0);
        let mut x = [0f32; PADDED];
        for (j, (frame, st)) in frames.iter().zip(states.iter_mut()).enumerate() {
            assert_eq!(frame.len(), FRAME, "Silero frames are {FRAME} samples");
            x[..CONTEXT].copy_from_slice(&st.context);
            x[CONTEXT..INPUT].copy_from_slice(frame);
            for i in 0..PADDED - INPUT {
                x[INPUT + i] = x[INPUT - 2 - i]; // reflect, edge excluded (torch/ONNX "reflect")
            }
            st.context.copy_from_slice(&x[INPUT - CONTEXT..INPUT]);
            for t in 0..FRAMES {
                s.cols[(j * FRAMES + t) * N_FFT..][..N_FFT]
                    .copy_from_slice(&x[t * HOP..t * HOP + N_FFT]);
            }
        }
        s.a.resize(n * FRAMES * 2 * BINS, 0.0);
        linear(
            &self.basis,
            None,
            2 * BINS,
            N_FFT,
            &s.cols,
            n * FRAMES,
            &mut s.a,
        );

        // Magnitude, laid out [n][bin][time] for the convolutions.
        s.b.resize(n * BINS * FRAMES, 0.0);
        for j in 0..n {
            for t in 0..FRAMES {
                let spec = &s.a[(j * FRAMES + t) * 2 * BINS..][..2 * BINS];
                for f in 0..BINS {
                    let (re, im) = (spec[f], spec[BINS + f]);
                    s.b[(j * BINS + f) * FRAMES + t] = (re * re + im * im).sqrt();
                }
            }
        }

        // Encoder: activations ping-pong between b (input) and a (output), [n][channel][time].
        let mut time = FRAMES;
        for conv in &self.convs {
            let tout = (time + 2 - 3) / conv.stride + 1;
            let k = conv.cin * 3;
            s.cols.resize(n * tout * k, 0.0);
            for j in 0..n {
                for t in 0..tout {
                    let col = &mut s.cols[(j * tout + t) * k..][..k];
                    for i in 0..conv.cin {
                        let row = &s.b[(j * conv.cin + i) * time..][..time];
                        for kk in 0..3 {
                            let p = (t * conv.stride + kk) as isize - 1;
                            col[i * 3 + kk] = if p >= 0 && (p as usize) < time {
                                row[p as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
            s.a.resize(n * tout * conv.cout, 0.0);
            linear(
                &conv.w,
                Some(&conv.b),
                conv.cout,
                k,
                &s.cols,
                n * tout,
                &mut s.a,
            );
            s.b.resize(n * conv.cout * tout, 0.0);
            for j in 0..n {
                for t in 0..tout {
                    for o in 0..conv.cout {
                        s.b[(j * conv.cout + o) * tout + t] =
                            s.a[(j * tout + t) * conv.cout + o].max(0.0);
                    }
                }
            }
            time = tout;
        }
        debug_assert_eq!(time, 1);

        // LSTMCell on the single remaining time step; s.b is now [n][128].
        s.hs.resize(n * HIDDEN, 0.0);
        for (j, st) in states.iter().enumerate() {
            s.hs[j * HIDDEN..][..HIDDEN].copy_from_slice(&st.h);
        }
        s.gx.resize(n * 4 * HIDDEN, 0.0);
        s.gh.resize(n * 4 * HIDDEN, 0.0);
        linear(
            &self.w_ih,
            Some(&self.b_ih),
            4 * HIDDEN,
            HIDDEN,
            &s.b[..n * HIDDEN],
            n,
            &mut s.gx,
        );
        linear(
            &self.w_hh,
            Some(&self.b_hh),
            4 * HIDDEN,
            HIDDEN,
            &s.hs,
            n,
            &mut s.gh,
        );
        for (j, st) in states.iter_mut().enumerate() {
            let gx = &s.gx[j * 4 * HIDDEN..][..4 * HIDDEN];
            let gh = &s.gh[j * 4 * HIDDEN..][..4 * HIDDEN];
            let mut z = 0f32;
            for u in 0..HIDDEN {
                let i = sigmoid(gx[u] + gh[u]);
                let f = sigmoid(gx[HIDDEN + u] + gh[HIDDEN + u]);
                let g = (gx[2 * HIDDEN + u] + gh[2 * HIDDEN + u]).tanh();
                let o = sigmoid(gx[3 * HIDDEN + u] + gh[3 * HIDDEN + u]);
                let c = f * st.c[u] + i * g;
                let h = o * c.tanh();
                st.c[u] = c;
                st.h[u] = h;
            }
            for u in 0..HIDDEN {
                z += self.dec_w[u] * st.h[u].max(0.0);
            }
            out[j] = sigmoid(z + self.dec_b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_that_are_not_the_graph_are_a_model_error() {
        assert!(matches!(
            SileroModel::from_onnx_bytes(b"not a model"),
            Err(TurnDetectionError::Model { .. })
        ));
    }
}
