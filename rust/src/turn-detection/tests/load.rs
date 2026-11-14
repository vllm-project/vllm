// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Model loading and the Smart Turn input window. Parity of the forward passes with
//! ONNX Runtime needs the real model files and is tested from Python
//! (`tests/entrypoints/speech_to_text/realtime/test_turn_detection.py`).

use std::io::Write;

use vllm_turn_detection::TurnDetectionError;
use vllm_turn_detection::pins;
use vllm_turn_detection::silero::SileroModel;
use vllm_turn_detection::smart_turn::{SAMPLES, SmartTurnModel, window};

#[test]
fn load_refuses_a_file_that_is_not_the_pinned_one() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"not a model").unwrap();
    match SileroModel::load(file.path()) {
        Err(TurnDetectionError::Checksum { expected, .. }) => {
            assert_eq!(expected, pins::SILERO_SHA256)
        }
        other => panic!("expected a checksum error, got {:?}", other.err()),
    }
    match SmartTurnModel::load(file.path()) {
        Err(TurnDetectionError::Checksum { expected, .. }) => {
            assert_eq!(expected, pins::SMART_TURN_SHA256)
        }
        other => panic!("expected a checksum error, got {:?}", other.err()),
    }
}

#[test]
fn load_reports_a_missing_file() {
    let missing = std::path::Path::new("/nonexistent/silero_vad.onnx");
    assert!(matches!(
        SileroModel::load(missing),
        Err(TurnDetectionError::Io { .. })
    ));
}

#[test]
fn window_pads_short_audio_in_front_and_keeps_the_last_8_s() {
    let short: Vec<f32> = (1..=10).map(|v| v as f32).collect();
    let w = window(&short);
    assert_eq!(w.len(), SAMPLES);
    assert!(w[..SAMPLES - 10].iter().all(|&v| v == 0.0));
    assert_eq!(&w[SAMPLES - 10..], &short[..]);
    let long: Vec<f32> = (0..SAMPLES + 5).map(|v| v as f32).collect();
    assert_eq!(window(&long), long[5..].to_vec());
}
