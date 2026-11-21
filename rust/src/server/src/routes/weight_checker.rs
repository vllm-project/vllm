// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, invalid_request};
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WeightCheckerAction {
    Checksum,
    Reset,
    Compare,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum WeightCheckerModel {
    #[default]
    Target,
    Draft,
    Both,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WeightCheckerRequest {
    action: WeightCheckerAction,
    #[serde(default)]
    model: WeightCheckerModel,
    #[serde(default)]
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

    if body.action == WeightCheckerAction::Reset {
        for validate_only in [true, false] {
            let errors: Vec<Option<String>> = client
                .call_utility("reset_weights", (body.model, validate_only))
                .await
                .map_err(|error| utility_call_error("reset_weights", error))?;
            if let Some(message) = errors.into_iter().flatten().next() {
                return Err(ApiError::Conflict { message });
            }
        }
        return Ok(Json(WeightCheckerResponse::Reset { status: "reset" }));
    }
    if body.action == WeightCheckerAction::Compare && body.baseline.is_none() {
        return Err(invalid_request!(
            param = Some("baseline"),
            "action='compare' requires a 'baseline' object"
        ));
    }

    let results: Vec<Vec<WorkerChecksums>> = client
        .call_utility("compute_weight_checksums", (body.model,))
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

    let Some(baseline) = body.baseline.filter(|_| body.action == WeightCheckerAction::Compare)
    else {
        return Ok(Json(WeightCheckerResponse::Checksums { checksums }));
    };
    let mismatches: Vec<String> = checksums
        .keys()
        .chain(baseline.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| checksums.get(*key) != baseline.get(*key))
        .cloned()
        .collect();
    Ok(Json(WeightCheckerResponse::Compare {
        matched: mismatches.is_empty(),
        mismatches,
    }))
}
