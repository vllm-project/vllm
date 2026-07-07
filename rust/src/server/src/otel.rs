//! OpenTelemetry trace-exporter bootstrap: a gRPC OTLP exporter (insecure by
//! default) feeding a batch span processor on a `TracerProvider` tagged with the
//! `vllm.*` resource attributes. No span is emitted here; this module only stands
//! up the provider and flushes it on shutdown.

use std::borrow::Cow;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use opentelemetry::KeyValue;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tokio::time::Instant;
use tracing::{info, warn};

use crate::config::ObservabilityConfig;

/// Env var selecting the OTLP transport.
const OTEL_TRACES_PROTOCOL_ENV: &str = "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL";

/// Instrumenting-module name set on every span's resource.
const INSTRUMENTING_MODULE_NAME: &str = "vllm.llm_engine";

/// Minimum flush budget, so a span buffered right before shutdown survives even
/// when `shutdown_timeout` is 0 (the default).
const MIN_FLUSH: Duration = Duration::from_secs(3);

/// Validate the OTLP transport selected by the environment, failing closed.
pub(crate) fn validate_protocol_env() -> Result<()> {
    validate_protocol(std::env::var(OTEL_TRACES_PROTOCOL_ENV).ok().as_deref())
}

/// Validate an OTLP transport value, failing closed. Only gRPC is wired;
/// `http/protobuf` is recognized but unimplemented, anything else rejected.
fn validate_protocol(protocol: Option<&str>) -> Result<()> {
    match protocol {
        None | Some("grpc") => Ok(()),
        Some("http/protobuf") => bail!(
            "OTLP protocol 'http/protobuf' is not yet supported in the Rust frontend; \
             use 'grpc' (the default) via OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
        ),
        Some(other) => bail!("unsupported OTLP protocol '{other}' is configured"),
    }
}

/// Default a scheme-less endpoint to plaintext `http://`, so a bare `host:port`
/// is accepted.
fn normalize_endpoint(endpoint: &str) -> Cow<'_, str> {
    if endpoint.contains("://") {
        Cow::Borrowed(endpoint)
    } else {
        Cow::Owned(format!("http://{endpoint}"))
    }
}

/// Build the OTLP trace provider and register it globally; returns the flush
/// handle, or `None` when no endpoint is set. Protocol is assumed already validated.
pub(crate) fn init_tracer_provider(obs: &ObservabilityConfig) -> Result<Option<SdkTracerProvider>> {
    let Some(endpoint) = obs.otlp_traces_endpoint.as_deref() else {
        return Ok(None);
    };

    let endpoint_url = normalize_endpoint(endpoint);
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint_url.as_ref())
        .build()
        .context("failed to build OTLP gRPC span exporter")?;

    let resource = Resource::builder()
        .with_attributes([
            KeyValue::new("vllm.instrumenting_module_name", INSTRUMENTING_MODULE_NAME),
            KeyValue::new("vllm.process_id", std::process::id().to_string()),
        ])
        .build();

    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_batch_exporter(exporter)
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());
    info!(endpoint, "OpenTelemetry trace exporter initialized (gRPC)");
    Ok(Some(provider))
}

/// Flush and shut down the provider before `deadline` (floored by `MIN_FLUSH`), so
/// a span buffered right before shutdown still ships. The calls block, so they run
/// on a blocking task.
pub(crate) async fn shutdown_provider(provider: SdkTracerProvider, deadline: Instant) {
    let budget = deadline.saturating_duration_since(Instant::now()).max(MIN_FLUSH);
    let flush = tokio::task::spawn_blocking(move || {
        if let Err(error) = provider.force_flush() {
            warn!(%error, "failed to flush OpenTelemetry spans on shutdown");
        }
        if let Err(error) = provider.shutdown() {
            warn!(%error, "failed to shut down OpenTelemetry provider");
        }
    });
    if tokio::time::timeout(budget, flush).await.is_err() {
        warn!(
            ?budget,
            "OpenTelemetry flush did not finish within the shutdown budget"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{ObservabilityConfig, init_tracer_provider, normalize_endpoint, validate_protocol};

    #[test]
    fn no_provider_when_tracing_disabled() {
        assert!(init_tracer_provider(&ObservabilityConfig::default()).unwrap().is_none());
    }

    #[test]
    fn endpoint_gets_a_scheme_only_when_missing() {
        assert_eq!(
            normalize_endpoint("localhost:4317"),
            "http://localhost:4317"
        );
        assert_eq!(normalize_endpoint("http://x:4317"), "http://x:4317");
        assert_eq!(normalize_endpoint("https://x:4317"), "https://x:4317");
    }

    #[test]
    fn protocol_defaults_and_grpc_are_accepted() {
        validate_protocol(None).unwrap();
        validate_protocol(Some("grpc")).unwrap();
    }

    #[test]
    fn http_protobuf_fails_closed_with_guidance() {
        let err = validate_protocol(Some("http/protobuf")).unwrap_err().to_string();
        assert!(err.contains("not yet supported"), "{err}");
    }

    #[test]
    fn unknown_protocol_is_rejected() {
        let err = validate_protocol(Some("carrier-pigeon")).unwrap_err().to_string();
        assert!(err.contains("unsupported OTLP protocol"), "{err}");
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use std::time::Duration;

    use opentelemetry::global;
    use opentelemetry::trace::Tracer as _;
    use opentelemetry_proto::tonic::collector::trace::v1::trace_service_server::{
        TraceService, TraceServiceServer,
    };
    use opentelemetry_proto::tonic::collector::trace::v1::{
        ExportTraceServiceRequest, ExportTraceServiceResponse,
    };
    use tokio::sync::mpsc;
    use tokio::time::Instant;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::{Request, Response, Status};

    use super::{ObservabilityConfig, init_tracer_provider, shutdown_provider};

    /// In-process OTLP gRPC collector that forwards each export over a channel for
    /// assertions.
    struct FakeCollector {
        exports: mpsc::UnboundedSender<ExportTraceServiceRequest>,
    }

    #[tonic::async_trait]
    impl TraceService for FakeCollector {
        async fn export(
            &self,
            request: Request<ExportTraceServiceRequest>,
        ) -> Result<Response<ExportTraceServiceResponse>, Status> {
            let _ = self.exports.send(request.into_inner());
            Ok(Response::new(ExportTraceServiceResponse::default()))
        }
    }

    /// The shutdown flush delivers a buffered span with the `vllm.*` resource attrs.
    #[tokio::test(flavor = "multi_thread")]
    #[serial_test::serial]
    async fn shutdown_flush_delivers_a_buffered_span_to_the_collector() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(TraceServiceServer::new(FakeCollector { exports: tx }))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
        });

        let obs = ObservabilityConfig {
            otlp_traces_endpoint: Some(format!("http://{addr}")),
            collect_detailed_traces: None,
        };
        let provider = init_tracer_provider(&obs).unwrap().expect("provider initialized");

        // Emit and end a span but do not flush it; the shutdown flush must deliver it.
        global::tracer("test").start("llm_request");

        shutdown_provider(provider, Instant::now() + Duration::from_secs(10)).await;

        let export = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("collector received an export within the budget")
            .expect("export payload present");
        let resource = export.resource_spans[0].resource.as_ref().expect("resource present");
        let keys: Vec<&str> = resource.attributes.iter().map(|kv| kv.key.as_str()).collect();
        assert!(keys.contains(&"vllm.instrumenting_module_name"), "{keys:?}");
        assert!(keys.contains(&"vllm.process_id"), "{keys:?}");

        server.abort();
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial_test::serial]
    async fn shutdown_flush_returns_within_budget_when_endpoint_is_unreachable() {
        let obs = ObservabilityConfig {
            // Reserved port 1: connection is refused.
            otlp_traces_endpoint: Some("http://127.0.0.1:1".to_string()),
            collect_detailed_traces: None,
        };
        let provider = init_tracer_provider(&obs).unwrap().expect("provider initialized");
        global::tracer("test").start("llm_request");

        let start = Instant::now();
        shutdown_provider(provider, start + Duration::from_secs(2)).await;
        assert!(
            start.elapsed() < Duration::from_secs(9),
            "flush must not hang"
        );
    }
}
