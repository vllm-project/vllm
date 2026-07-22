// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use thiserror_ext::AsReport as _;
use tonic::{Request, Response, Status};

use super::{ControlServer, pb};
use crate::state::AppState;

pub(crate) type ControlGrpcService = ControlServer<ControlServiceImpl>;

/// gRPC control service backed by the shared application state.
pub struct ControlServiceImpl {
    state: Arc<AppState>,
}

impl ControlServiceImpl {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl pb::control_server::Control for ControlServiceImpl {
    async fn abort(
        &self,
        request: Request<pb::AbortRequest>,
    ) -> Result<Response<pb::AbortResponse>, Status> {
        let request_ids = request.into_inner().request_ids;
        if request_ids.is_empty() {
            return Ok(Response::new(pb::AbortResponse {}));
        }
        self.state
            .chat
            .abort(&request_ids)
            .await
            .map_err(|error| Status::internal(error.to_report_string()))?;
        Ok(Response::new(pb::AbortResponse {}))
    }
}
