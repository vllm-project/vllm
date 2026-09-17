// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

#[derive(Debug, Error, PartialEq)]
pub enum SamplingParamsError {
    #[error("{parameter} must be a finite number, got {value}")]
    NotFinite { parameter: &'static str, value: f32 },
    #[error("{parameter} must be in {expected}, got {value}")]
    OutOfRange {
        parameter: &'static str,
        value: f32,
        expected: &'static str,
    },
}

fn validate_frequency_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("frequency_penalty", value, -2.0, 2.0, "[-2, 2]")
}

fn validate_presence_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("presence_penalty", value, -2.0, 2.0, "[-2, 2]")
}

fn validate_temperature(value: f32) -> Result<(), SamplingParamsError> {
    validate_finite("temperature", value)?;
    validate_closed_range("temperature", value, 0.0, 2.0, "[0, 2]")
}

fn validate_top_p(value: f32) -> Result<(), SamplingParamsError> {
    if value > 0.0 && value <= 1.0 {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter: "top_p",
        value,
        expected: "(0, 1]",
    })
}

fn validate_min_p(value: f32) -> Result<(), SamplingParamsError> {
    validate_closed_range("min_p", value, 0.0, 1.0, "[0, 1]")
}

fn validate_repetition_penalty(value: f32) -> Result<(), SamplingParamsError> {
    validate_finite("repetition_penalty", value)?;
    if value > 0.0 {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter: "repetition_penalty",
        value,
        expected: "(0, inf)",
    })
}

pub(crate) fn validate_resolved_sampling_params(
    params: &EngineCoreSamplingParams,
) -> Result<(), SamplingParamsError> {
    validate_temperature(params.temperature)?;
    validate_top_p(params.top_p)?;
    validate_min_p(params.min_p)?;
    validate_frequency_penalty(params.frequency_penalty)?;
    validate_presence_penalty(params.presence_penalty)?;
    validate_repetition_penalty(params.repetition_penalty)
}

fn validate_finite(parameter: &'static str, value: f32) -> Result<(), SamplingParamsError> {
    if value.is_finite() {
        return Ok(());
    }
    Err(SamplingParamsError::NotFinite { parameter, value })
}

fn validate_closed_range(
    parameter: &'static str,
    value: f32,
    min: f32,
    max: f32,
    expected: &'static str,
) -> Result<(), SamplingParamsError> {
    if value >= min && value <= max {
        return Ok(());
    }
    Err(SamplingParamsError::OutOfRange {
        parameter,
        value,
        expected,
    })
}
