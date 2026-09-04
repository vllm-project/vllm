// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::server_info::ServerInfoConfigFormat;
use crate::state::AppState;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConfigFormat {
    Text,
    Json,
}

impl From<ConfigFormat> for ServerInfoConfigFormat {
    fn from(value: ConfigFormat) -> Self {
        match value {
            ConfigFormat::Text => Self::Text,
            ConfigFormat::Json => Self::Json,
        }
    }
}

fn default_config_format() -> ConfigFormat {
    ConfigFormat::Text
}

#[derive(Debug, Deserialize)]
pub(crate) struct ServerInfoParams {
    #[serde(default = "default_config_format")]
    config_format: ConfigFormat,
}

/// Get server configuration and environment metadata.
pub async fn server_info(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ServerInfoParams>,
) -> Response {
    match state.server_info_response(params.config_format.into()) {
        Some(response) => Json(response).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
