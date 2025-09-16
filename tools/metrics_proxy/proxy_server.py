"""FastAPI application that proxies OpenAI compatible chat completions and logs metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Mapping, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .metrics_recorder import (ProxyMetricsRecorder, RequestOutcome,
                               RequestSummary)

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


@dataclass
class ProxyConfig:
    """Runtime configuration for the metrics proxy."""

    upstream_url: str
    model_name: str
    engine_label: str = "proxy"
    max_concurrency: int = 16
    max_model_len: int = 4096
    connect_timeout: float = 10.0
    read_timeout: float = 600.0
    write_timeout: float = 600.0
    request_timeout: float = 600.0
    expose_metrics_endpoint: bool = True

    def normalized_upstream(self) -> str:
        return self.upstream_url.rstrip("/")


def create_app(config: ProxyConfig) -> FastAPI:
    """Create a FastAPI application configured to proxy chat completions."""

    app = FastAPI(title="vLLM Metrics Proxy")
    metrics = ProxyMetricsRecorder(model_name=config.model_name,
                                   engine_label=config.engine_label,
                                   max_model_len=config.max_model_len)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    app.state.metrics = metrics
    app.state.semaphore = semaphore
    app.state.config = config
    app.state.client = None

    @app.on_event("startup")
    async def _startup() -> None:
        timeout = httpx.Timeout(timeout=config.request_timeout,
                                connect=config.connect_timeout,
                                read=config.read_timeout,
                                write=config.write_timeout,
                                pool=None)
        app.state.client = httpx.AsyncClient(
            base_url=config.normalized_upstream(),
            timeout=timeout,
            follow_redirects=True,
        )
        logger.info("Metrics proxy started – forwarding to %s", config.upstream_url)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        client: Optional[httpx.AsyncClient] = app.state.client
        if client is not None:
            await client.aclose()
        logger.info("Metrics proxy stopped")

    @app.post(CHAT_COMPLETIONS_PATH)
    async def chat_completions(request: Request) -> Response:
        client: httpx.AsyncClient = app.state.client
        if client is None:
            raise HTTPException(status_code=503, detail="Proxy not initialised")

        metrics: ProxyMetricsRecorder = app.state.metrics
        semaphore: asyncio.Semaphore = app.state.semaphore

        req_id = uuid.uuid4().hex[:8]
        arrival = time.perf_counter()
        metrics.increment_waiting()
        acquired = False
        try:
            await semaphore.acquire()
            acquired = True
        finally:
            metrics.decrement_waiting()
        if not acquired:
            raise HTTPException(status_code=503, detail="Unable to acquire proxy slot")

        queue_time = max(time.perf_counter() - arrival, 0.0)
        metrics.observe_queue_time(queue_time)

        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            semaphore.release()
            logger.warning("[%s] Invalid JSON payload: %s", req_id, exc)
            raise HTTPException(status_code=400,
                                detail="Request body must be valid JSON") from exc

        stream = bool(payload.get("stream", False))
        n_param = _coerce_int(payload.get("n", 1), default=1)
        max_tokens = _extract_max_tokens(payload)

        headers = _filter_request_headers(request.headers)

        metrics.increment_running()
        start_forward = time.perf_counter()

        if stream:
            response = await _handle_streaming_request(req_id, client, payload,
                                                        headers, metrics, semaphore,
                                                        queue_time, start_forward,
                                                        n_param, max_tokens)
            return response

        try:
            upstream_response = await client.post(CHAT_COMPLETIONS_PATH,
                                                  json=payload,
                                                  headers=headers)
        except httpx.HTTPError as exc:
            metrics.decrement_running()
            semaphore.release()
            logger.exception("[%s] Upstream request failed: %s", req_id, exc)
            raise HTTPException(status_code=502,
                                detail="Failed to contact upstream server") from exc

        inference_time = max(time.perf_counter() - start_forward, 0.0)
        data: Optional[Dict[str, Any]] = None
        if upstream_response.headers.get("content-type", "").startswith(
                "application/json"):
            try:
                data = upstream_response.json()
            except ValueError:
                logger.warning("[%s] Failed to decode JSON response", req_id)
        prompt_tokens, completion_tokens, total_tokens = _parse_usage(
            data.get("usage") if isinstance(data, dict) else None)
        finish_reasons = _extract_finish_reasons(data)
        success = upstream_response.is_success

        outcome = RequestOutcome(queue_time=queue_time,
                                 inference_time=inference_time,
                                 ttft=None,
                                 prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens,
                                 total_tokens=total_tokens,
                                 finish_reasons=finish_reasons,
                                 max_tokens=max_tokens,
                                 n=n_param,
                                 success=success,
                                 status_code=upstream_response.status_code,
                                 stream=False)
        summary = metrics.finalize_request(outcome)
        metrics.decrement_running()
        semaphore.release()
        _log_request_summary(req_id, summary)

        return Response(content=upstream_response.content,
                        status_code=upstream_response.status_code,
                        headers=_filter_response_headers(upstream_response.headers),
                        media_type=upstream_response.headers.get("content-type"))

    @app.get("/internal/metrics")
    async def internal_metrics() -> Dict[str, List[Dict[str, Any]]]:
        return metrics.snapshot()

    if config.expose_metrics_endpoint:
        @app.get("/metrics")
        async def prometheus_metrics() -> PlainTextResponse:
            body = metrics.render_prometheus()
            return PlainTextResponse(content=body,
                                     media_type="text/plain; version=0.0.4")

    @app.get("/internal/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    return app


async def _handle_streaming_request(
    req_id: str,
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Mapping[str, str],
    metrics: ProxyMetricsRecorder,
    semaphore: asyncio.Semaphore,
    queue_time: float,
    start_forward: float,
    n_param: int,
    max_tokens: Optional[int],
) -> StreamingResponse:
    upstream_cm = client.stream("POST", CHAT_COMPLETIONS_PATH, json=payload,
                                headers=headers)
    try:
        upstream_response = await upstream_cm.__aenter__()
    except httpx.HTTPError as exc:
        metrics.decrement_running()
        semaphore.release()
        logger.exception("[%s] Upstream streaming request failed: %s", req_id, exc)
        raise HTTPException(status_code=502,
                            detail="Failed to contact upstream server") from exc

    finish_reasons: Dict[int, str] = {}
    usage_payload: Optional[Dict[str, Any]] = None
    buffer = ""
    first_token_time: Optional[float] = None

    async def event_generator() -> AsyncGenerator[bytes, None]:
        nonlocal buffer, usage_payload, first_token_time
        try:
            async for chunk in upstream_response.aiter_bytes():
                if chunk:
                    now = time.perf_counter()
                    text = chunk.decode("utf-8", errors="ignore")
                    buffer += text
                    buffer, usage_payload, first_token_time = _process_sse_buffer(
                        buffer, now, finish_reasons, usage_payload,
                        first_token_time)
                    yield chunk
        finally:
            # Process any remaining buffered data
            if buffer:
                buffer, usage_payload, first_token_time = _process_sse_buffer(
                    buffer, time.perf_counter(), finish_reasons,
                    usage_payload, first_token_time)
            inference_time = max(time.perf_counter() - start_forward, 0.0)
            ttft = None
            if first_token_time is not None:
                ttft = max(first_token_time - start_forward, 0.0)
            prompt_tokens, completion_tokens, total_tokens = _parse_usage(
                usage_payload)
            outcome = RequestOutcome(
                queue_time=queue_time,
                inference_time=inference_time,
                ttft=ttft,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reasons=list(finish_reasons.values()),
                max_tokens=max_tokens,
                n=n_param,
                success=upstream_response.is_success,
                status_code=upstream_response.status_code,
                stream=True,
            )
            summary = metrics.finalize_request(outcome)
            metrics.decrement_running()
            semaphore.release()
            await upstream_cm.__aexit__(None, None, None)
            _log_request_summary(req_id, summary)

    headers = _filter_response_headers(upstream_response.headers)
    media_type = upstream_response.headers.get("content-type", "text/event-stream")
    return StreamingResponse(event_generator(),
                             status_code=upstream_response.status_code,
                             headers=headers,
                             media_type=media_type)


def _process_sse_buffer(
    buffer: str,
    event_time: float,
    finish_reasons: Dict[int, str],
    usage_payload: Optional[Dict[str, Any]],
    first_token_time: Optional[float],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[float]]:
    while True:
        if "\n" not in buffer:
            return buffer, usage_payload, first_token_time
        line, buffer = buffer.split("\n", 1)
        line = line.rstrip("\r")
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Failed to decode SSE payload: %s", data)
            continue
        usage_payload, first_token_time = _handle_streaming_payload(
            payload, event_time, finish_reasons, usage_payload,
            first_token_time)
    return buffer, usage_payload, first_token_time


def _handle_streaming_payload(
    payload: Dict[str, Any],
    event_time: float,
    finish_reasons: Dict[int, str],
    usage_payload: Optional[Dict[str, Any]],
    first_token_time: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    if "usage" in payload and isinstance(payload["usage"], dict):
        usage_payload = payload["usage"]

    for choice in payload.get("choices", []):
        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        index = _coerce_int(choice.get("index", 0), default=0)
        if finish_reason:
            finish_reasons[index] = finish_reason
        delta = choice.get("delta") or {}
        if delta:
            if any(delta.get(key) for key in ("content", "text")):
                first_token_time = event_time if first_token_time is None else first_token_time
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                first_token_time = event_time if first_token_time is None else first_token_time

    return usage_payload, first_token_time


def _parse_usage(usage: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not usage or not isinstance(usage, dict):
        return None, None, None
    prompt = _coerce_int(usage.get("prompt_tokens"))
    completion = _coerce_int(usage.get("completion_tokens"))
    total = _coerce_int(usage.get("total_tokens"))
    return prompt, completion, total


def _extract_finish_reasons(data: Optional[Dict[str, Any]]) -> List[str]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list):
        return []
    reasons: List[str] = []
    for choice in choices:
        if isinstance(choice, dict) and choice.get("finish_reason"):
            reasons.append(choice["finish_reason"])
    return reasons


def _extract_max_tokens(payload: Mapping[str, Any]) -> Optional[int]:
    for key in ("max_completion_tokens", "max_tokens", "max_new_tokens",
                "max_output_tokens"):
        if key in payload:
            value = _coerce_int(payload.get(key))
            if value is not None:
                return value
    return None


def _coerce_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _filter_request_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    excluded = {"host", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded}


def _filter_response_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    excluded = {"content-length", "transfer-encoding", "connection"}
    filtered = {k: v for k, v in headers.items() if k.lower() not in excluded}
    # Streaming responses expect this header to be present.
    if "content-type" not in {k.lower() for k in filtered}:
        filtered["content-type"] = "text/event-stream"
    return filtered


def _log_request_summary(req_id: str, summary: RequestSummary) -> None:
    fields = [
        f"status={summary.status_code}",
        f"success={summary.success}",
        f"queue={summary.queue_time:.3f}s",
        f"inference={summary.inference_time:.3f}s",
        f"e2e={summary.e2e_time:.3f}s",
    ]
    if summary.ttft is not None:
        fields.append(f"ttft={summary.ttft:.3f}s")
    if summary.inter_token_latency is not None:
        fields.append(f"avg_token_latency={summary.inter_token_latency:.3f}s")
    if summary.prompt_tokens is not None:
        fields.append(f"prompt_tokens={summary.prompt_tokens}")
    if summary.completion_tokens is not None:
        fields.append(f"completion_tokens={summary.completion_tokens}")
    if summary.max_tokens is not None:
        fields.append(f"max_tokens={summary.max_tokens}")
    if summary.finish_reasons:
        fields.append(f"finish_reasons={summary.finish_reasons}")
    logger.info("Proxy metrics [%s]: %s", req_id, ", ".join(fields))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vLLM metrics proxy.")
    parser.add_argument("--upstream-url", required=True,
                        help="Base URL of the upstream OpenAI compatible server")
    parser.add_argument("--model-name", required=True,
                        help="Name of the served model (used for metrics labels)")
    parser.add_argument("--engine-label", default="proxy",
                        help="Engine label attached to emitted metrics")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Listen address for the proxy server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Listen port for the proxy server")
    parser.add_argument("--max-concurrency", type=int, default=16,
                        help="Maximum number of concurrent upstream requests")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Model context length used for histogram buckets")
    parser.add_argument("--connect-timeout", type=float, default=10.0,
                        help="Connection timeout when contacting upstream")
    parser.add_argument("--read-timeout", type=float, default=600.0,
                        help="Read timeout when contacting upstream")
    parser.add_argument("--write-timeout", type=float, default=600.0,
                        help="Write timeout when contacting upstream")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="Overall request timeout for upstream operations")
    parser.add_argument("--disable-metrics-endpoint", action="store_true",
                        help="Do not expose the /metrics Prometheus endpoint")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ...)")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    config = ProxyConfig(upstream_url=args.upstream_url,
                         model_name=args.model_name,
                         engine_label=args.engine_label,
                         max_concurrency=args.max_concurrency,
                         max_model_len=args.max_model_len,
                         connect_timeout=args.connect_timeout,
                         read_timeout=args.read_timeout,
                         write_timeout=args.write_timeout,
                         request_timeout=args.request_timeout,
                         expose_metrics_endpoint=not args.disable_metrics_endpoint)

    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
