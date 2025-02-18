# Setup OpenTelemetry POC

1. Install OpenTelemetry packages:

    ```console
    pip install \
      'opentelemetry-sdk>=1.26.0,<1.27.0' \
      'opentelemetry-api>=1.26.0,<1.27.0' \
      'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' \
      'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'
    ```

1. Start Jaeger in a docker container:

    ```console
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

    ```console
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    ```

    Then set vLLM's service name for OpenTelemetry, enable insecure connections to Jaeger and run vLLM:

    ```console
    export OTEL_SERVICE_NAME="vllm-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ```

1. In a new shell, send requests with trace context from a dummy client

    ```console
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

```console
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://$JAEGER_IP:4318/v1/traces
vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```

## Instrumentation of FastAPI

OpenTelemetry allows automatic instrumentation of FastAPI.

1. Install the instrumentation library

    ```console
    pip install opentelemetry-instrumentation-fastapi
    ```

1. Run vLLM with `opentelemetry-instrument`

    ```console
    opentelemetry-instrument vllm serve facebook/opt-125m
    ```

1. Send a request to vLLM and find its trace in Jaeger. It should contain spans from FastAPI.

![FastAPI Spans](https://i.imgur.com/hywvoOJ.png)
