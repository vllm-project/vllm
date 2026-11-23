# TTFT Forecast

vLLM can forecast how long a request arriving now would wait for its first
token, as a function of the prompt tokens the engine would have to compute for
it. A router in front of several engines reads the forecast of each one and
sends the request where it would start soonest, without keeping its own model
of engine speed.

The forecast is off by default.

## Enabling

Set `VLLM_TTFT_FORECAST=1` when starting the server:

```bash
VLLM_TTFT_FORECAST=1 vllm serve Qwen/Qwen3-0.6B
```

The scheduler then times its own steps and keeps a snapshot of the requests it
holds: the steps in flight, the prompt tokens each running request has left,
and the waiting queue, with a queued prompt counted at the length its cached
prefix leaves to compute. A forecast replays the scheduling rule (token budget,
`--max-num-seqs`, `--long-prefill-token-threshold`) on that snapshot with a
probe request appended. Nothing is configured.

The first forecast is published once the engine has timed eight steps.

## Reading the Forecast

A forecast is a curve: seconds to first token over new tokens, the prompt
tokens that are not covered by a cached prefix. There is one curve per
cached-prefix length, because the tokens of the prefix are context for the new
ones.

### `GET /forecast`

```bash
curl 'http://localhost:8000/forecast?cached=0,4096'
```

`cached` is a comma-separated list of up to 8 cached-prefix lengths (default
`0`). The response holds, for each length:

- `curves`: segments `[u0, t0, u1, t1]`. A request with `u` new tokens,
  `u0 < u <= u1`, is forecast at `t0 + (t1 - t0) * (u - u0) / (u1 - u0)` seconds.
- `floor`: the same curve for this engine with nothing queued.

and for the engine:

- `ready`: whether a forecast is published.
- `age`: seconds since the scheduler state behind the forecast was captured.
- `restore_seconds`: recent time a request waited for a KV connector to load
  its prefix before it could be scheduled.
- `busy_share`: share of recent time the engine had a step in flight.
- `self_check`: the error of the engine's own recent forecasts against the
  first tokens it then produced.

`POST /forecast` takes `{"cached": [...], "queries": [[new_tokens,
cached_tokens], ...]}` with up to 256 queries and returns one value per query
in `ttft`.

The endpoint answers on the engine's input thread, so a step in progress does
not delay it. Like `/metrics`, it is served without an API key and reports
timing and queue sizes only.

### Metrics

The same curves are published in `/metrics` at fixed knots, so a router that
already scrapes the engine needs no extra call:

- `vllm:ttft_forecast_seconds{new_tokens, cached_tokens}` and
  `vllm:ttft_forecast_idle_seconds{new_tokens, cached_tokens}`: the curve and
  the idle-engine curve. `new_tokens` is at powers of two and on both sides of
  the first eight step boundaries (multiples of `--max-num-batched-tokens`);
  `cached_tokens` is at 0, a quarter and half of the model length. Interpolate
  linearly between knots.
- `vllm:ttft_forecast_restore_seconds`, `vllm:ttft_forecast_busy_share`.
- `vllm:ttft_forecast_age_seconds`: seconds since the gauges were computed.

The gauges are refreshed at each scrape, at most once per 20 ms. A scrape
waits at most 0.1 s for the engine: past that it is served the last values and
`vllm:ttft_forecast_age_seconds` grows until the engine answers again.

## Using It in a Router

For each engine, evaluate the curve at the number of the request's prompt
tokens that the engine does not have cached. A value that is `age` seconds old
is moved forward as `max(idle, forecast - age)`: queued work drains as time
passes, the request's own compute does not. For a prefix the router places in
a KV connector tier, add `restore_seconds`.

## Limitations

- Requests that arrive after a forecast are not in it.
- The replay covers the default scheduler with chunked prefill. Priority
  scheduling and preemption are not replayed.
- The cached-prefix lookup sees device blocks only. A queued request whose
  prefix a KV connector would load counts at its uncached length until the
  scheduler reaches it.
- Time in the API server before a request's arrival is stamped is outside the
  forecast.
- Each engine forecasts its own stage; with disaggregated prefill and decode
  the two forecasts add up.
- The gauges and the endpoint are served by the Python API server.

## Debugging

`VLLM_TTFT_FORECAST_DEBUG=1` adds `detail` to the `/forecast` response: the
last 16 arrivals with their forecast, the measured time to first token and the
scheduler state the forecast was made from.
