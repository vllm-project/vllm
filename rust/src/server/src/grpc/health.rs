// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tonic::server::NamedService;
use tonic_health::ServingStatus;
use tonic_health::server::HealthReporter;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineCoreClient;

use super::{ControlGrpcService, InferenceGrpcService};

/// Health service for liveness probes. It reports `NOT_SERVING` only after an
/// engine failure: pause, sleep, weight updates, and orderly shutdown leave it
/// `SERVING` until the listener closes.
pub(crate) const LIVENESS_SERVICE: &str = "liveness";

/// Health service for readiness probes. Like `vllm.Inference`, it reports
/// whether the engines can accept and process new requests, and
/// `NOT_SERVING` on shutdown or engine failure.
pub(crate) const READINESS_SERVICE: &str = "readiness";

/// The engine client signals that the gRPC health services follow.
pub(crate) struct EngineHealth {
    /// `false` once the engines fail, which is permanent.
    pub liveness: watch::Receiver<bool>,
    /// Whether the engines can accept and process new requests.
    pub readiness: watch::Receiver<bool>,
}

impl EngineHealth {
    pub fn subscribe(client: &EngineCoreClient) -> Self {
        Self {
            liveness: client.subscribe_health(),
            readiness: client.subscribe_readiness(),
        }
    }
}

/// Publishes the gRPC health services from [`EngineHealth`].
pub(crate) struct HealthMonitor {
    reporter: HealthReporter,
    engine: EngineHealth,
    /// Readiness as last published to `readiness` and `vllm.Inference`.
    ready: bool,
}

impl HealthMonitor {
    /// Register every named health service with the engines' current
    /// readiness. Call before the listener accepts connections so clients
    /// never see an unregistered service.
    pub async fn register(reporter: HealthReporter, mut engine: EngineHealth) -> Self {
        let ready = *engine.readiness.borrow_and_update();
        reporter.set_serving::<ControlGrpcService>().await;
        reporter.set_service_status(LIVENESS_SERVICE, ServingStatus::Serving).await;
        set_readiness(&reporter, ready).await;
        Self {
            reporter,
            engine,
            ready,
        }
    }

    /// Keep the health services in line with the engines until shutdown.
    ///
    /// `readiness` and `vllm.Inference` follow the engines' readiness. On
    /// shutdown, every service except `liveness` reports `NOT_SERVING`; on
    /// engine failure, `liveness` does too.
    pub async fn run(mut self, shutdown: CancellationToken) {
        let inference_service = InferenceGrpcService::NAME;
        let control_service = ControlGrpcService::NAME;
        let status = ServingStatus::NotServing;
        let mut readiness_open = true;
        let engine_failed = loop {
            let ready = tokio::select! {
                result = self.engine.liveness.wait_for(|healthy| !*healthy) => {
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
                    break true;
                }
                _ = shutdown.cancelled() => {
                    info!(
                        status = ?status,
                        reason = "server_shutdown",
                        "server shutting down; marking gRPC health services except liveness as not serving"
                    );
                    break false;
                }
                changed = self.engine.readiness.changed(), if readiness_open => {
                    readiness_open = changed.is_ok();
                    *self.engine.readiness.borrow_and_update()
                }
            };
            // Changes can coalesce back to the published value.
            if ready != self.ready {
                self.ready = ready;
                debug!(ready, "engine readiness changed");
                set_readiness(&self.reporter, ready).await;
            }
        };

        let reporter = &mut self.reporter;
        set_readiness(reporter, false).await;
        reporter.set_not_serving::<ControlGrpcService>().await;
        // Both gRPC services use the same engine client, so overall server health
        // mirrors their shared engine health.
        reporter.set_service_status("", status).await;

        if engine_failed {
            reporter.set_service_status(LIVENESS_SERVICE, status).await;
            shutdown.cancelled().await;
            info!(
                reason = "server_shutdown",
                "server shutting down; closing gRPC health watches"
            );
        }

        reporter.clear_service_status(inference_service).await;
        reporter.clear_service_status(control_service).await;
        reporter.clear_service_status("").await;
        reporter.clear_service_status(LIVENESS_SERVICE).await;
        reporter.clear_service_status(READINESS_SERVICE).await;
    }
}

async fn set_readiness(reporter: &HealthReporter, ready: bool) {
    let status = if ready {
        ServingStatus::Serving
    } else {
        ServingStatus::NotServing
    };
    for service in [READINESS_SERVICE, InferenceGrpcService::NAME] {
        reporter.set_service_status(service, status).await;
    }
}
