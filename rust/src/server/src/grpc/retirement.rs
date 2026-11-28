// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::time::Duration;

use thiserror_ext::AsReport as _;
use tonic::Status;

use crate::state::AppState;

/// Internal IDs captured while their generation streams still own the mapping.
pub(super) struct RequestRetirement(Vec<String>);

impl RequestRetirement {
    pub fn capture_generation(state: &AppState, external_id: &str) -> Result<Self, Status> {
        let retirement = Self::capture(state, &[external_id.to_owned()])?;
        if retirement.0.len() != 1 {
            return Err(Status::failed_precondition(
                "generation retirement requires a unique active external request ID",
            ));
        }
        Ok(retirement)
    }

    pub fn capture(state: &AppState, external_ids: &[String]) -> Result<Self, Status> {
        state
            .engine_core_llm()
            .and_then(|llm| llm.request_ids_for_retirement(external_ids))
            .map(Self)
            .ok_or_else(|| {
                Status::failed_precondition(
                    "retirement requires active request IDs and engine-core request-ID randomization",
                )
            })
    }

    pub async fn confirm(&self, state: &AppState) -> Result<(), Status> {
        tokio::time::timeout(
            Duration::from_secs(30),
            state.engine_core_client().abort_and_confirm(&self.0),
        )
        .await
        .map_err(|_| Status::deadline_exceeded("scheduler retirement was not confirmed"))?
        .map_err(|error| Status::unavailable(error.to_report_string()))
    }
}
