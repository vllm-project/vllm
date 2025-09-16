# vLLM Metrics Proxy Tools

This directory contains a lightweight proxy and helper utilities that mirror vLLM's
Prometheus metrics for OpenAI-compatible `/v1/chat/completions` traffic. The proxy sits
in front of an existing vLLM (or any OpenAI-compatible) deployment, forwards requests to
its upstream server, and records metrics such as queue time, time-to-first-token (TTFT),
number of running/waiting requests, token counts, and success/error totals. The metrics
can be scraped in Prometheus format or consumed as structured JSON for custom tooling.

## Installation

The proxy depends on FastAPI, httpx, uvicorn, and prometheus-client. Install the project
in editable mode (recommended when working inside the repository) and include the
runtime dependencies:

```bash
pip install -e .[dev]
# or install only the required packages
pip install fastapi uvicorn httpx prometheus-client
```

## Running the proxy server

Launch the proxy by pointing it at an existing `/v1/chat/completions` endpoint and
supplying the model name you want attached to emitted metrics:

```bash
python -m tools.metrics_proxy.proxy_server \
  --upstream-url http://localhost:8001 \
  --model-name my-model-name
```

Key flags you can tweak:

* `--host` / `--port`: listening interface for the proxy (default `0.0.0.0:8000`).
* `--engine-label`: value recorded in the `engine` Prometheus label (default `proxy`).
* `--max-concurrency`: number of concurrent upstream requests the proxy allows before
  new arrivals wait in the queue.
* `--max-model-len`: maximum context window used to size latency/token histograms.
* `--connect-timeout`, `--read-timeout`, `--write-timeout`, `--request-timeout`: HTTP
  timeouts applied to upstream calls.
* `--disable-metrics-endpoint`: omit the Prometheus `/metrics` endpoint when you only
  need JSON snapshots.
* `--log-level`: adjust proxy logging verbosity.

The proxy exposes the following routes:

* `POST /v1/chat/completions`: forwards regular or streaming chat completions to the
  upstream server while tracking queue/inference durations and token usage.
* `GET /internal/metrics`: returns a JSON snapshot of all counters, gauges, histograms,
  and vectors maintained by the proxy (`ProxyMetricsRecorder.snapshot()`).
* `GET /metrics`: (optional) renders metrics in Prometheus exposition format for a
  scraper such as Prometheus or Grafana Agent.
* `GET /internal/healthz`: simple health probe that returns `{ "status": "ok" }`.

Each request generates an INFO-level summary log including queue time, inference time,
TTFT (for streaming responses), and token usage to ease manual inspection.

## Inspecting metrics from the command line

For quick local debugging, the `print_metrics` helper fetches and pretty-prints the JSON
metrics snapshot:

```bash
python -m tools.metrics_proxy.print_metrics \
  --url http://localhost:8000/internal/metrics
```

This prints each gauge, counter, vector, and histogram with its labels and values so you
can verify the proxy is tracking the expected statistics.

## Embedding the recorder directly

If you are embedding the proxy logic inside another service, you can import
`ProxyMetricsRecorder` from `tools.metrics_proxy` and wire it into your own FastAPI app
or request pipeline. The recorder mirrors the metrics defined in `vllm.v1.metrics`
(`Gauge`, `Counter`, `Histogram`, and vector types) and provides helpers such as
`increment_waiting`, `observe_queue_time`, and `finalize_request` to update counters in
line with the native vLLM server implementation.
