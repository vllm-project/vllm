// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use tokio::sync::watch;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tonic::server::NamedService;
use tonic_health::ServingStatus;
use tonic_health::server::HealthReporter;
use tracing::{debug, info, warn};
use vllm_llm::Llm;

use super::{ControlGrpcService, InferenceGrpcService, RenderGrpcService};
use crate::state::AppState;

/// Health service for liveness probes. It reports `NOT_SERVING` only after an
/// engine failure: pause, sleep, weight updates, and orderly shutdown leave it
/// `SERVING` until the listener closes.
pub(crate) const LIVENESS_SERVICE: &str = "liveness";

/// Health service for readiness probes. Like `vllm.Inference`, it reports
/// whether the engines can accept and process new requests, and
/// `NOT_SERVING` on shutdown or engine failure.
pub(crate) const READINESS_SERVICE: &str = "readiness";

/// The [`Llm`] signals that the gRPC health services follow.
pub(crate) struct EngineHealth {
    /// `false` once the engines fail, which is permanent.
    pub liveness: watch::Receiver<bool>,
    /// Whether the engines can accept and process new requests.
    pub readiness: watch::Receiver<bool>,
}

impl EngineHealth {
    pub fn subscribe(llm: &dyn Llm) -> Self {
        Self {
            liveness: llm.subscribe_health(),
            readiness: llm.subscribe_readiness(),
        }
    }
}

/// The name of `vllm.Render`, the same for either server state.
const RENDER_SERVICE: &str = RenderGrpcService::<AppState>::NAME;

/// What a server's gRPC health services follow.
pub(crate) enum HealthSource {
    /// The engines behind the main server.
    Engines(EngineHealth),
    /// None: the engine-free render server, whose services, including
    /// `liveness` and `readiness`, serve until shutdown.
    EngineFree,
}

impl HealthSource {
    /// Services that serve until shutdown.
    fn services(&self) -> &'static [&'static str] {
        match self {
            Self::Engines(_) => &[ControlGrpcService::NAME, RENDER_SERVICE],
            Self::EngineFree => &[RENDER_SERVICE],
        }
    }

    /// Services that follow the engines' readiness until shutdown.
    fn readiness_services(&self) -> &'static [&'static str] {
        match self {
            Self::Engines(_) => &[READINESS_SERVICE, InferenceGrpcService::NAME],
            Self::EngineFree => &[READINESS_SERVICE],
        }
    }
}

/// Publishes the gRPC health services from a [`HealthSource`].
pub(crate) struct HealthMonitor {
    reporter: HealthReporter,
    source: HealthSource,
    /// Readiness as last published to the readiness services.
    ready: bool,
}

impl HealthMonitor {
    /// Register every named health service with the engines' current
    /// readiness. Call before the listener accepts connections so clients
    /// never see an unregistered service.
    pub async fn register(reporter: HealthReporter, mut source: HealthSource) -> Self {
        let ready = match &mut source {
            HealthSource::Engines(engine) => *engine.readiness.borrow_and_update(),
            HealthSource::EngineFree => true,
        };
        for service in source.services() {
            reporter.set_service_status(service, ServingStatus::Serving).await;
        }
        reporter.set_service_status(LIVENESS_SERVICE, ServingStatus::Serving).await;
        set_readiness(&reporter, source.readiness_services(), ready).await;
        Self {
            reporter,
            source,
            ready,
        }
    }

    /// Run a gRPC server until shutdown, keeping the health services in line
    /// with the engines.
    ///
    /// `server` builds the server from the signal that stops its listener. On
    /// shutdown, every service except `liveness` reports `NOT_SERVING` before
    /// that signal fires and before `force` can abort the server, so watchers
    /// see the withdrawal before their connection closes. Returns `None` if
    /// `force` aborted the server.
    pub async fn serve<S: Future>(
        self,
        shutdown: CancellationToken,
        force: impl Future<Output = ()>,
        server: impl FnOnce(WaitForCancellationFutureOwned) -> S,
    ) -> Option<S::Output> {
        // Cancelled by `run` once the withdrawal is published.
        let withdrawn = CancellationToken::new();
        let server = server(withdrawn.clone().cancelled_owned());
        let server = async {
            tokio::select! {
                output = server => Some(output),
                // `force` fires with `shutdown` under `--shutdown-timeout 0`.
                () = async { force.await; withdrawn.cancelled().await } => None,
            }
        };
        // The graceful drain ends once `run` removes the services, ending
        // their watches.
        let (output, ()) = tokio::join!(server, self.run(shutdown, &withdrawn));
        output
    }

    /// Keep the health services in line with the engines until shutdown,
    /// cancel `withdrawn`, then remove the services, which ends their watches.
    ///
    /// `readiness` and `vllm.Inference` follow the engines' readiness. On
    /// shutdown, every service except `liveness` reports `NOT_SERVING`; on
    /// engine failure, `liveness` does too.
    async fn run(mut self, shutdown: CancellationToken, withdrawn: &CancellationToken) {
        let status = ServingStatus::NotServing;
        let readiness_services = self.source.readiness_services();
        let mut readiness_open = true;
        let engine = match &mut self.source {
            HealthSource::Engines(engine) => engine,
            HealthSource::EngineFree => {
                shutdown.cancelled().await;
                info!(
                    status = ?status,
                    reason = "server_shutdown",
                    "server shutting down; marking gRPC health services except liveness as not serving"
                );
                return self.withdraw(shutdown, withdrawn, false).await;
            }
        };
        let engine_failed = loop {
            let ready = tokio::select! {
                // A failure can arrive with the shutdown it causes; `liveness`
                // must still report it.
                biased;

                result = engine.liveness.wait_for(|healthy| !*healthy) => {
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
                changed = engine.readiness.changed(), if readiness_open => {
                    readiness_open = changed.is_ok();
                    *engine.readiness.borrow_and_update()
                }
            };
            // Changes can coalesce back to the published value.
            if ready != self.ready {
                self.ready = ready;
                debug!(ready, "engine readiness changed");
                set_readiness(&self.reporter, readiness_services, ready).await;
            }
        };
        self.withdraw(shutdown, withdrawn, engine_failed).await;
    }

    /// Publish `NOT_SERVING`, cancel `withdrawn` once the server may stop, then
    /// remove the services, which ends their watches.
    async fn withdraw(
        mut self,
        shutdown: CancellationToken,
        withdrawn: &CancellationToken,
        engine_failed: bool,
    ) {
        let status = ServingStatus::NotServing;
        let services = self.source.services();
        let readiness_services = self.source.readiness_services();
        let reporter = &mut self.reporter;
        set_readiness(reporter, readiness_services, false).await;
        for service in services {
            reporter.set_service_status(service, status).await;
        }
        // The gRPC services are served alongside the same engine client, so overall
        // server health mirrors its engine health.
        reporter.set_service_status("", status).await;

        if engine_failed {
            reporter.set_service_status(LIVENESS_SERVICE, status).await;
            shutdown.cancelled().await;
            info!(
                reason = "server_shutdown",
                "server shutting down; closing gRPC health watches"
            );
        }
        // Published, so the server may stop.
        withdrawn.cancel();

        for service in readiness_services.iter().chain(services) {
            reporter.clear_service_status(service).await;
        }
        reporter.clear_service_status("").await;
        reporter.clear_service_status(LIVENESS_SERVICE).await;
    }
}

async fn set_readiness(reporter: &HealthReporter, services: &[&str], ready: bool) {
    let status = if ready {
        ServingStatus::Serving
    } else {
        ServingStatus::NotServing
    };
    for service in services {
        reporter.set_service_status(service, status).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures::{FutureExt as _, StreamExt as _};
    use tonic::Request;
    use tonic_health::pb::HealthCheckRequest;
    use tonic_health::pb::health_server::Health as _;
    use tonic_health::server::HealthService;

    use super::*;

    /// A monitor for ready, live engines, a way to read its statuses, and the
    /// engine signals, which must outlive the test: a closed channel reads as
    /// an engine failure.
    async fn monitor() -> (HealthMonitor, HealthService, [watch::Sender<bool>; 2]) {
        let (reporter, _) = tonic_health::server::health_reporter();
        let health = HealthService::from_health_reporter(reporter.clone());
        let (liveness_tx, liveness) = watch::channel(true);
        let (readiness_tx, readiness) = watch::channel(true);
        let engine = EngineHealth {
            liveness,
            readiness,
        };
        let monitor = HealthMonitor::register(reporter, HealthSource::Engines(engine)).await;
        (monitor, health, [liveness_tx, readiness_tx])
    }

    /// Whether `readiness` reports `SERVING`, read without yielding. By the
    /// time the listener stops, the service is `NOT_SERVING` or removed.
    fn serving(health: &HealthService) -> bool {
        status(health, READINESS_SERVICE) == Some(ServingStatus::Serving as i32)
    }

    /// The status of `service`, read without yielding; `None` if unregistered.
    fn status(health: &HealthService, service: &str) -> Option<i32> {
        let request = Request::new(HealthCheckRequest {
            service: service.to_string(),
        });
        let response = health.check(request).now_or_never().expect("uncontended read");
        response.ok().map(|response| response.get_ref().status)
    }

    /// The render server registers only its own service besides the probes,
    /// serving until shutdown withdraws them before the listener stops.
    #[tokio::test]
    async fn engine_free_services_serve_until_shutdown() {
        let (reporter, _) = tonic_health::server::health_reporter();
        let health = HealthService::from_health_reporter(reporter.clone());
        let monitor = HealthMonitor::register(reporter, HealthSource::EngineFree).await;
        for service in [RENDER_SERVICE, LIVENESS_SERVICE, READINESS_SERVICE, ""] {
            assert_eq!(
                status(&health, service),
                Some(ServingStatus::Serving as i32),
                "{service}"
            );
        }
        for service in [InferenceGrpcService::NAME, ControlGrpcService::NAME] {
            assert_eq!(status(&health, service), None, "{service}");
        }

        let shutdown = CancellationToken::new();
        shutdown.cancel();
        let at_stop = monitor
            .serve(shutdown, std::future::pending(), |stop| async move {
                stop.await;
                [RENDER_SERVICE, READINESS_SERVICE, ""].map(|service| status(&health, service))
            })
            .await
            .expect("not aborted");
        // `NOT_SERVING`, or already removed.
        for status in at_stop {
            assert_ne!(status, Some(ServingStatus::Serving as i32));
        }
    }

    #[tokio::test]
    async fn listener_stops_after_readiness_is_withdrawn() {
        let (monitor, health, _engine) = monitor().await;
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        let serving_at_stop = monitor
            .serve(shutdown, std::future::pending(), |stop| async move {
                stop.await;
                serving(&health)
            })
            .await;

        assert_eq!(serving_at_stop, Some(false));
    }

    #[tokio::test]
    async fn liveness_reports_failure_that_arrives_with_shutdown() {
        let (monitor, health, engine) = monitor().await;
        let liveness = health
            .watch(Request::new(HealthCheckRequest {
                service: LIVENESS_SERVICE.to_string(),
            }))
            .await
            .unwrap()
            .into_inner();
        engine[0].send(false).unwrap();
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        monitor.serve(shutdown, std::future::pending(), |stop| stop).await.unwrap();

        // The watch ends once the service is removed, after its last status.
        let statuses: Vec<_> = liveness.map(|response| response.unwrap().status).collect().await;
        assert_eq!(statuses.last(), Some(&(ServingStatus::NotServing as i32)));
    }

    #[tokio::test]
    async fn forced_abort_waits_until_readiness_is_withdrawn() {
        /// Records readiness when the server is dropped, as an abort does.
        struct Server {
            health: HealthService,
            serving_at_drop: Arc<Mutex<Option<bool>>>,
        }
        impl Drop for Server {
            fn drop(&mut self) {
                *self.serving_at_drop.lock().unwrap() = Some(serving(&self.health));
            }
        }

        let (monitor, health, _engine) = monitor().await;
        let serving_at_drop = Arc::new(Mutex::new(None));
        let server = Server {
            health,
            serving_at_drop: serving_at_drop.clone(),
        };
        let shutdown = CancellationToken::new();
        shutdown.cancel();

        let output = monitor
            .serve(shutdown, async {}, |_stop| async move {
                let _server = server;
                std::future::pending::<()>().await
            })
            .await;

        assert_eq!(output, None);
        assert_eq!(*serving_at_drop.lock().unwrap(), Some(false));
    }
}
