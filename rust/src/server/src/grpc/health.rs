// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::server::NamedService;
use tonic_health::ServingStatus;
use tonic_health::server::HealthReporter;
use tracing::{info, warn};

use super::{ControlGrpcService, GenerateGrpcService};

pub(crate) async fn monitor_health(
    mut health_reporter: HealthReporter,
    mut engine_health: watch::Receiver<bool>,
    shutdown: CancellationToken,
) {
    let generate_service = GenerateGrpcService::NAME;
    let control_service = ControlGrpcService::NAME;
    let status = ServingStatus::NotServing;
    let health_event_first = tokio::select! {
        result = engine_health.wait_for(|healthy| !*healthy) => {
            match result {
                Ok(_) => warn!(
                    status = ?status,
                    reason = "engine_unhealthy",
                    "marking gRPC health services as not serving"
                ),
                Err(error) => warn!(
                    %error,
                    status = ?status,
                    reason = "health_channel_closed",
                    "engine health channel closed; marking gRPC health services as not serving"
                ),
            }
            true
        }
        _ = shutdown.cancelled() => {
            info!(
                status = ?status,
                reason = "server_shutdown",
                "server shutting down; marking gRPC health services as not serving"
            );
            false
        }
    };

    health_reporter.set_not_serving::<GenerateGrpcService>().await;
    health_reporter.set_not_serving::<ControlGrpcService>().await;
    // Both gRPC services use the same engine client, so overall server health
    // mirrors their shared engine health.
    health_reporter.set_service_status("", status).await;

    if health_event_first {
        shutdown.cancelled().await;
        info!(
            reason = "server_shutdown",
            "server shutting down; closing gRPC health watches"
        );
    }

    health_reporter.clear_service_status(generate_service).await;
    health_reporter.clear_service_status(control_service).await;
    health_reporter.clear_service_status("").await;
}
