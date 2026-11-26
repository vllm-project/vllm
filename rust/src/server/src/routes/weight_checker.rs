// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;
use vllm_engine_core_client::EngineCoreClient;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WeightCheckerAction {
    Checksum,
    Reset,
    Compare,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WeightCheckerModel {
    #[default]
    Target,
    Draft,
    Both,
}

impl WeightCheckerModel {
    /// Name the Python worker expects; serde would encode the enum as a tuple.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Draft => "draft",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct WeightCheckerRequest {
    action: WeightCheckerAction,
    #[serde(default)]
    model: WeightCheckerModel,
    baseline: Option<BTreeMap<String, String>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WorkerChecksums {
    Checksums(BTreeMap<String, String>),
    Conflict(String),
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum WeightCheckerResponse {
    Checksums {
        checksums: BTreeMap<String, String>,
    },
    Reset {
        status: &'static str,
    },
    Compare {
        #[serde(rename = "match")]
        matched: bool,
        mismatches: Vec<String>,
    },
}

/// Checksum, reset, or compare model weights against a baseline.
pub async fn weight_checker(
    State(state): State<Arc<AppState>>,
    body: Result<Json<WeightCheckerRequest>, JsonRejection>,
) -> Result<Json<WeightCheckerResponse>, ApiError> {
    let Json(body) = body?;
    let client = state.engine_core_client();
    let response = match body.action {
        WeightCheckerAction::Checksum => WeightCheckerResponse::Checksums {
            checksums: weight_checksums(client, body.model).await?,
        },
        WeightCheckerAction::Compare => {
            let baseline = body.baseline.ok_or_else(|| {
                ApiError::invalid_body("action='compare' requires a 'baseline' object")
            })?;
            let mismatches = mismatches(&weight_checksums(client, body.model).await?, &baseline);
            WeightCheckerResponse::Compare {
                matched: mismatches.is_empty(),
                mismatches,
            }
        }
        WeightCheckerAction::Reset => {
            let errors: Vec<Option<String>> = client
                .call_utility("reset_weights", (body.model.as_str(),))
                .await
                .map_err(|error| utility_call_error("reset_weights", error))?;
            if let Some(message) = errors.into_iter().flatten().next() {
                return Err(ApiError::Conflict { message });
            }
            WeightCheckerResponse::Reset { status: "reset" }
        }
    };
    Ok(Json(response))
}

/// Merge every worker's weight checksums, rejecting duplicate keys.
pub(crate) async fn weight_checksums(
    client: &EngineCoreClient,
    model: WeightCheckerModel,
) -> Result<BTreeMap<String, String>, ApiError> {
    let results: Vec<Vec<WorkerChecksums>> = client
        .call_utility("compute_weight_checksums", (model.as_str(),))
        .await
        .map_err(|error| utility_call_error("compute_weight_checksums", error))?;
    let mut checksums = BTreeMap::new();
    for worker in results.into_iter().flatten() {
        let worker = match worker {
            WorkerChecksums::Checksums(checksums) => checksums,
            WorkerChecksums::Conflict(message) => return Err(ApiError::Conflict { message }),
        };
        for (key, digest) in worker {
            if checksums.contains_key(&key) {
                return Err(ApiError::server_error(format!(
                    "Duplicate weight checksum keys: {key}"
                )));
            }
            checksums.insert(key, digest);
        }
    }
    Ok(checksums)
}

/// Keys whose digest differs from the baseline or is missing on either side.
pub(crate) fn mismatches(
    checksums: &BTreeMap<String, String>,
    baseline: &BTreeMap<String, String>,
) -> Vec<String> {
    checksums
        .keys()
        .chain(baseline.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| checksums.get(*key) != baseline.get(*key))
        .cloned()
        .collect()
}
