# Tracing vLLM start up with OpenTelemetry

vLLM supports tracing through key cold start phases, as trace spans, leading up
to when the FastAPI HTTP server is running.

* As a vLLM user tracing can help inform vLLM configuration for faster start
  up, e.g. by changing settings such as `--enforce-eager`, `--load-format`.

* As a vLLM contributor you can use tracing to find areas worth optimizing and
  measure the impact of your changes.

Beware that the set of trace spans and their attributes are subject to change
and should not be considered stable, for now.

## Usage

1. Install required packages. vLLM does not install these by default.

    ```
    pip install opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp
    ```

2. Start an OpenTelemetry collector for vLLM to send traces to. See the
   [OpenTelemetry registry](https://opentelemetry.io/ecosystem/registry/?s=collector) for
    options. We'll use [jaeger](https://www.jaegertracing.io/) as an example.

    ```
    docker run \
      -p 16686:16686 \
      -p 4317:4317 \
      jaegertracing/jaeger:latest
    ```

3. Configure [OpenTelemetry environment variables](https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/) for vLLM

    ```
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://localhost:4317
    export OTEL_SERVICE_NAME="vllm.server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    ```

4. Start vLLM, it will enable tracing by default based on the environment variables.

    ```
    vllm serve google/gemma-3-4b-it
    ```

5. View the traces in http://localhost:16686/

## Extending trace coverage  

Read the [Python instrumentation guide](https://opentelemetry.io/docs/languages/python/instrumentation/) to learn more about tracing and for span/attribute naming see [naming conventions](https://opentelemetry.io/docs/specs/semconv/general/naming/).

OpenTelemetry is an optional vLLM dependency and consequently we concentrate anything requiring opentelemetry packages in `vllm/tracing.py` to limit the places where we need to do import checks.

For tracing start up we use a single trace that is propagated between the API server process and the engine core process.

To add new trace spans to a module use a global tracer instance (similar to logger) and use that to capture spans per the guide above.

```
from vllm.tracing import get_tracer
...
tracer = get_tracer(__name__)
```
