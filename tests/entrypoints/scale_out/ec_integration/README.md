# Scale-out EC connector e2e tests

End-to-end tests for the disaggregated multimodal (encoder-cache) flow over
the scale-out endpoints, in the bash-script style of
`tests/v1/kv_connector/nixl_integration`.

## Topology

```text
render (GPU-less)                encode (EC producer)
  /v1/chat/completions/render      /inference/v1/generate  (kwargs_data)
  /v1/chat/completions/derender                        |
                     prefill (EC consumer)
                       /inference/v1/generate  (mm_metadata + ec_transfer_params)
```

- The **render** server (`vllm launch render`) preprocesses the chat request
  into token ids and multimodal features (`kwargs_data` + `mm_metadata`).
- The **encode** instance (`ec_role: ec_producer`, encode-only) runs the
  vision encoder on `kwargs_data` and publishes embeddings through the
  `ECExampleConnector` shared storage. Its response carries
  `ec_transfer_params`.
- The **prefill** instance (`ec_role: ec_consumer`) receives metadata-only
  features (`mm_metadata`, no `kwargs_data`) plus `ec_transfer_params`, loads
  the embeddings through the EC connector, and generates output token ids.
- The **derender** endpoint turns the output token ids back into a chat
  completion response.

## Running

```bash
bash tests/entrypoints/scale_out/ec_integration/run_scale_out_ec_e2e_test.sh
```

The script first collects baseline outputs from a single `vllm serve`
instance, then runs the scale-out topology and compares the derendered
outputs for exact equality. It also checks that metadata-only features are
rejected when `ec_transfer_params` is missing.

Defaults: `Qwen/Qwen3-VL-2B-Instruct`, encode on GPU 0, prefill on GPU 1.
Override via environment variables (`MODEL`, `GPU_E`, `GPU_PD`,
`EC_SHARED_STORAGE_PATH`, `PYTHON=.venv/bin/python`, ...); see the header of
the script for the full list.
