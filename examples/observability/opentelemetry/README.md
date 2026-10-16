# Setup OpenTelemetry POC

> **Note:** The core OpenTelemetry packages (`opentelemetry-sdk`, `opentelemetry-api`, `opentelemetry-exporter-otlp`, `opentelemetry-semantic-conventions-ai`) are bundled with vLLM. Manual installation is not required.

1. Start Jaeger in a docker container:

    ```bash
    # From: https://www.jaegertracing.io/docs/1.57/getting-started/
    docker run --rm --name jaeger \
        -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
        -p 6831:6831/udp \
        -p 6832:6832/udp \
        -p 5778:5778 \
        -p 16686:16686 \
        -p 4317:4317 \
        -p 4318:4318 \
        -p 14250:14250 \
        -p 14268:14268 \
        -p 14269:14269 \
        -p 9411:9411 \
        jaegertracing/all-in-one:1.57
    ```

1. In a new shell, export Jaeger IP:

    ```bash
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    ```

    Then set vLLM's service name for OpenTelemetry, enable insecure connections to Jaeger and run vLLM:

    ```bash
    export OTEL_SERVICE_NAME="vllm-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ```

1. In a new shell, send requests with trace context from a dummy client

    ```bash
    export JAEGER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    export OTEL_SERVICE_NAME="client-service"
    python dummy_client.py
    ```

1. Open Jaeger webui: <http://localhost:16686/>

    In the search pane, select `vllm-server` service and hit `Find Traces`. You should get a list of traces, one for each request.
    ![Traces](https://i.imgur.com/GYHhFjo.png)

1. Clicking on a trace will show its spans and their tags. In this demo, each trace has 2 spans. One from the dummy client containing the prompt text and one from vLLM containing metadata about the request.
![Spans details](https://i.imgur.com/OPf6CBL.png)

## Exporter Protocol

OpenTelemetry supports either `grpc` or `http/protobuf` as the transport protocol for trace data in the exporter.
By default, `grpc` is used. To set `http/protobuf` as the protocol, configure the `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` environment variable as follows:

```bash
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://$JAEGER_IP:4318/v1/traces
vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```

## Inference spans

With `--otlp-traces-endpoint` configured, `llm_request` spans are created at
frontend engine admission using the request arrival timestamp. Tool and non-tool
requests use the same lifecycle: non-streaming API spans finish after response
conversion, while streaming API spans finish after streaming and generator cleanup.
Errors, cancellation, and disconnects also close the spans. Direct engine callers
without an API owner finish at engine completion; Responses background work has
its own owner and can outlive the initial HTTP response.

Multiple engine requests, such as parallel choices, retain separate spans.
Existing `gen_ai.latency.*` attributes still measure engine timings; span duration
can be longer because it includes API response processing. Requests that never
reach engine admission do not produce an inference span.

### Tool-calling attributes

Chat Completions and Messages requests get the following attributes when they
explicitly supply non-null `tools`, `tool_choice`, or `parallel_tool_calls`.
Other requests still produce ordinary inference spans, without these attributes.
Presence does not require a nonempty tools list: for example, Chat
`tool_choice="none"` without tools qualifies. API validation is unchanged:
Chat rejects `tools=[]`, while Messages accepts it.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `vllm.tool_calling.present` | Boolean | Tool-related request fields are present. |
| `vllm.tool_calling.tool_count` | Integer | Number of received tool definitions. |
| `vllm.tool_calling.tool_choice` | String | `unset`, `auto`, `none`, `required`, `named`, or `other`. |
| `vllm.tool_calling.api_surface` | String | `chat_completions` or `messages`. |
| `vllm.tool_calling.response_has_tool_calls` | Boolean | The response count is greater than zero, including partial streams. |
| `vllm.tool_calling.tool_calls_count` | Integer | Tool-call objects included in server output. |
| `vllm.tool_calling.parser_status` | String | Response outcome, defined below. |

Request summaries are captured before parser mutation or Messages-to-Chat
conversion. Choices are normalized from the validated native request; Messages
`any` becomes `required` and `tool` becomes `named`. Counts reflect filtered Chat
choices or converted Messages `tool_use` blocks, not raw parser output. Repeated
streaming argument fragments count once per tool call. These are request-level
totals repeated on each owned inference span, not per-choice counts to sum.

| `parser_status` | Meaning |
| --- | --- |
| `success` | Response processing completed with at least one tool call. |
| `no_tool_calls` | Response processing completed without tool calls. |
| `parse_failure` | A parsing exception surfaced, including invalid Messages tool arguments during conversion. |
| `interrupted` | Response processing did not complete normally, for example after an engine error, cancellation, or disconnect. |

Interrupted streams retain counts already emitted; they are not reported as
successful. A surfaced `parse_failure` takes precedence over interruption.
Errors swallowed by a parser cannot be detected by these attributes. Output
counts do not prove client receipt, tool execution, argument correctness, or
preservation of fields by an upstream router.

Summaries are attached only to recording spans and follow the configured trace
sampler. Unsampled requests do not export tool observations. Prometheus counters
remain independent of sampling and keep their existing eligibility rules.
Trace propagation remains limited to W3C `traceparent` and `tracestate`; tool
metadata is not carried in headers. The summary attributes include no tool names,
schemas, arguments, prompt/response text, exception text, or content-derived hashes.

## Instrumentation of FastAPI

OpenTelemetry allows automatic instrumentation of FastAPI.

1. Install the instrumentation library

    ```bash
    pip install opentelemetry-instrumentation-fastapi
    ```

1. Run vLLM with `opentelemetry-instrument`

    ```bash
    opentelemetry-instrument vllm serve facebook/opt-125m
    ```

1. Send a request to vLLM and find its trace in Jaeger. It should contain spans from FastAPI.

![FastAPI Spans](https://i.imgur.com/hywvoOJ.png)
