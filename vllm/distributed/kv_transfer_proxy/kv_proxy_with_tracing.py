import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status, StatusCode

# Initialize the tracer provider with a resource containing the service name
resource = Resource(
    attributes={ResourceAttributes.SERVICE_NAME: "pd-sep-vllm_proxy"})

trace.set_tracer_provider(TracerProvider(resource=resource))

# Configure the OTLP exporter to send data to Jaeger
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)

# Add the BatchSpanProcessor to the tracer provider
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter))

# Set the global propagator to TraceContext
set_global_textmap(TraceContextTextMapPropagator())

# Initialize the FastAPI app
app = FastAPI()

# Instrument the FastAPI app to automatically create spans for incoming HTTP requests
FastAPIInstrumentor.instrument_app(app)

# Obtain a tracer instance
tracer = trace.get_tracer(__name__)

# Base URLs for the two vLLM processes (set to the root of the API)
VLLM_1_BASE_URL = "http://localhost:8000/v1"
VLLM_2_BASE_URL = "http://localhost:8001/v1"

# Initialize variables to hold the persistent clients
app.state.vllm1_client = None
app.state.vllm2_client = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize persistent HTTPX clients for vLLM services on startup.
    """
    HTTPXClientInstrumentor().instrument()
    app.state.vllm2_client = httpx.AsyncClient(timeout=None,
                                               base_url=VLLM_2_BASE_URL)
    app.state.vllm1_client = httpx.AsyncClient(timeout=None,
                                               base_url=VLLM_1_BASE_URL)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Close the persistent HTTPX clients on shutdown.
    """
    await app.state.vllm1_client.aclose()
    await app.state.vllm2_client.aclose()


async def send_request_to_vllm(client: httpx.AsyncClient, req_data: dict):
    """
    Send a request to a vLLM process using a persistent client.
    """
    response = await client.post("/completions",
                                 json=req_data)  # Correct endpoint path
    response.raise_for_status()
    return response


async def stream_vllm_response(client: httpx.AsyncClient, req_data: dict):
    """
    Asynchronously stream the response from a vLLM process using a persistent client.

    Args:
        client (httpx.AsyncClient): The persistent HTTPX client.
        req_data (dict): The JSON payload to send.

    Yields:
        bytes: Chunks of the response data.
    """
    async with client.stream(
            "POST", "/completions",
            json=req_data) as response:  # Correct endpoint path
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/completions")
async def proxy_request(request: Request):
    """
    Proxy endpoint that forwards requests to two vLLM services.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        StreamingResponse: The streamed response from the second vLLM service.
    """
    req_data = await request.json()

    with tracer.start_as_current_span("proxy-request-span") as proxy_span:
        proxy_span.set_attribute("http.method", request.method)
        proxy_span.set_attribute("http.url", str(request.url))
        proxy_span.set_attribute("client.ip", request.client.host)

        # Fire-and-forget request to vLLM-1
        with tracer.start_as_current_span(
                "send-to-prefill-vllm") as send_vllm1_span:
            send_vllm1_span.set_attribute("vllm.url",
                                          VLLM_1_BASE_URL + "/completions")
            try:
                # Use asyncio.create_task to avoid waiting for the response
                asyncio.create_task(
                    send_request_to_vllm(app.state.vllm1_client, req_data))
                send_vllm1_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                send_vllm1_span.record_exception(e)
                send_vllm1_span.set_status(Status(StatusCode.ERROR, str(e)))

        # Proceed to vLLM-2 immediately
        with tracer.start_as_current_span(
                "send-to-decode-vllm") as send_vllm2_span:
            send_vllm2_span.set_attribute("vllm.url",
                                          VLLM_2_BASE_URL + "/completions")
            try:

                async def generate_stream():
                    with tracer.start_as_current_span(
                            "stream-vllm2-response") as stream_span:
                        async for chunk in stream_vllm_response(
                                app.state.vllm2_client, req_data):
                            stream_span.add_event("Streaming chunk")
                            yield chunk

                return StreamingResponse(generate_stream(),
                                         media_type="application/json")
            except Exception as e:
                send_vllm2_span.record_exception(e)
                send_vllm2_span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
