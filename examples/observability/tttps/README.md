# TTTPS Proof-of-Time Provenance Middleware

[TTTPS](https://github.com/Helm-Protocol/OpenTTT) (OpenTTT) attaches a
cryptographic audit-trail timestamp and integrity hash to model output, so a
third party can later verify *when* a response was produced and that it
hasn't been altered since. It does not certify legal/regulatory compliance
(EU AI Act, FDA, etc.) or output correctness — it is a provenance receipt,
not a compliance stamp.

This example wires TTTPS into vLLM's OpenAI-compatible server using vLLM's
existing [`--middleware`](../../../docs/configuration/serve_args.md) CLI
flag — no changes to vLLM itself are required. `TTTPSMiddleware` is a
regular Starlette `BaseHTTPMiddleware`, the same extension point other
observability add-ons (see [`../opentelemetry`](../opentelemetry)) attach
through.

## Prerequisites

Get a free API key from the public Provenance API (no card required):

```bash
curl -X POST https://kpp.kenosian.com/v1/keys \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "use_case": "vllm middleware demo"}'
```

This returns a `kpp_prov_...` key with a starting quota of free seals.

## Run

```bash
export KPP_API_KEY=kpp_prov_...

vllm serve facebook/opt-125m \
  --middleware tttps_middleware.TTTPSMiddleware
```

(Run this from the `examples/observability/tttps` directory, or otherwise
make sure `tttps_middleware.py` is importable — e.g. by adding this
directory to `PYTHONPATH`.)

## Call it

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "facebook/opt-125m", "prompt": "Hello,", "max_tokens": 8}' \
  -D - -o /dev/null | grep -i x-tttps-receipt
```

The response includes an `X-TTTPS-Receipt` header, a JSON blob with a
`receipt_id` you can independently re-verify:

```bash
curl -X POST https://kpp.kenosian.com/v1/verify \
  -H "Content-Type: application/json" \
  -d '{"receipt_id": "<receipt_id from the header>"}'
```

## Notes

- **Fail-open**: any error talking to the Provenance API (timeout, quota
  exhaustion, network failure) degrades to a `{"status": "degraded", ...}`
  receipt. The underlying vLLM response body and status code are never
  affected.
- **Streaming (`stream=true`) is not covered** by this example.
  `BaseHTTPMiddleware` buffers non-streaming bodies only; anchoring a live
  SSE stream needs a different approach.
- Tested against a locally built vLLM using this exact `--middleware`
  invocation, confirming a live `X-TTTPS-Receipt` header on a real
  `/v1/completions` round trip.
