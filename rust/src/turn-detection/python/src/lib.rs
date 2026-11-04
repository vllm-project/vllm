// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python bindings for `vllm-turn-detection`, used by the `/v1/realtime` turn
//! detection (`vllm/entrypoints/speech_to_text/realtime/turn_detection.py`).
//!
//! Both models are shared across sessions and release the GIL while computing.
//! Audio arrives as contiguous float32 buffers (numpy arrays) at 16 kHz.
//!
//! The release profile aborts on panic, which would take the API server down,
//! so inputs are validated here before they reach the models' shape asserts.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use thiserror_ext::AsReport;
use vllm_turn_detection::silero::{FRAME, SileroModel, SileroScratch, SileroState};
use vllm_turn_detection::smart_turn::{LogMel, SmartTurnModel, window};
use vllm_turn_detection::{TurnDetectionError, pins};

fn to_py(e: TurnDetectionError) -> PyErr {
    match e {
        TurnDetectionError::Io { .. } => PyOSError::new_err(e.to_report_string()),
        _ => PyValueError::new_err(e.to_report_string()),
    }
}

fn f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let buf: PyBuffer<f32> = PyBuffer::get(obj)
        .map_err(|_| PyTypeError::new_err("expected a contiguous float32 buffer (numpy array)"))?;
    buf.to_vec(obj.py())
}

/// Per-stream Silero state (LSTM state and 64-sample context). Immutable:
/// `SileroVad.infer` returns the next one.
#[pyclass(frozen, module = "vllm._rust_turn_detection")]
#[derive(Clone)]
struct SileroVadState(SileroState);

thread_local! {
    static SCRATCH: RefCell<SileroScratch> = RefCell::new(SileroScratch::default());
}

/// Silero VAD v6.2 at 16 kHz, one 512-sample frame per call.
#[pyclass(frozen, module = "vllm._rust_turn_detection")]
struct SileroVad {
    model: Arc<SileroModel>,
}

#[pymethods]
impl SileroVad {
    /// Load the pinned `silero_vad.onnx`, checking its sha256.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model = SileroModel::load(Path::new(model_path)).map_err(to_py)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    #[classattr]
    fn frame_samples() -> usize {
        FRAME
    }

    fn new_state(&self) -> SileroVadState {
        SileroVadState(SileroState::default())
    }

    /// Speech probability of one 512-sample frame, and the stream's next state.
    fn infer(
        &self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        state: &Bound<'_, SileroVadState>,
    ) -> PyResult<(f32, SileroVadState)> {
        let frame = f32_vec(frame)?;
        if frame.len() != FRAME {
            return Err(PyValueError::new_err(format!(
                "a Silero frame has exactly {FRAME} samples, got {}",
                frame.len()
            )));
        }
        let mut state = state.get().0.clone();
        let model = self.model.clone();
        let (probability, state) = py.detach(move || {
            let p = SCRATCH.with(|s| model.infer(&frame, &mut state, &mut s.borrow_mut()));
            (p, state)
        });
        Ok((probability, SileroVadState(state)))
    }
}

/// Smart Turn v3.2 (fp32): probability that the speech ending at the end of the
/// audio is a complete turn.
#[pyclass(frozen, module = "vllm._rust_turn_detection")]
struct SmartTurn {
    model: Arc<SmartTurnModel>,
    mel: Arc<LogMel>,
}

#[pymethods]
impl SmartTurn {
    /// Load the pinned `smart-turn-v3.2-gpu.onnx`, checking its sha256.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model = SmartTurnModel::load(Path::new(model_path)).map_err(to_py)?;
        Ok(Self {
            model: Arc::new(model),
            mel: Arc::new(LogMel::new()),
        })
    }

    /// Score the last 8 s of `audio` (shorter audio is zero-padded in front).
    fn predict(&self, py: Python<'_>, audio: &Bound<'_, PyAny>) -> PyResult<f32> {
        let audio = f32_vec(audio)?;
        let (model, mel) = (self.model.clone(), self.mel.clone());
        Ok(py.detach(move || model.score(&mel.compute(&window(&audio)))))
    }
}

#[pymodule]
fn _rust_turn_detection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SileroVad>()?;
    m.add_class::<SileroVadState>()?;
    m.add_class::<SmartTurn>()?;
    for (name, value) in [
        ("SILERO_REPO", pins::SILERO_REPO),
        ("SILERO_REVISION", pins::SILERO_REVISION),
        ("SILERO_FILE", pins::SILERO_FILE),
        ("SMART_TURN_REPO", pins::SMART_TURN_REPO),
        ("SMART_TURN_REVISION", pins::SMART_TURN_REVISION),
        ("SMART_TURN_FILE", pins::SMART_TURN_FILE),
    ] {
        m.add(name, value)?;
    }
    Ok(())
}
