# Filesystem encoder-cache integration tests

Requires one CUDA GPU with enough memory for Qwen3-VL-8B-Instruct, an installed
vLLM checkout, and the repository's CUDA test dependencies. The default model is
the public Hugging Face checkpoint `Qwen/Qwen3-VL-8B-Instruct`. Set `MODEL` to
override it. The image is the existing repository asset `../hato.jpg`; no remote
image service is needed. Cache files use pytest's `tmp_path`.

Run from the repository root:

```bash
.venv/bin/python -m pytest -q -s \
  tests/v1/ec_connector/integration/one_card/test_filesystem_tiering.py
```

Each case starts a server, sends an image request, waits for a published `.ec`
entry, then restarts the server using the same cache directory. The restarted
server receives either one request or two concurrent requests for the same image.
The tests compare generated text and require CPU-to-device cache-load completion
logs. Prefix caching is disabled.

Concurrent HTTP submission does not guarantee that filesystem reads overlap.
The deterministic assertion that repeated requests submit only one pending
promotion is covered by
`test_secondary_hit_is_promoted_once_and_published_on_completion` in
`../../unit/tiering/test_manager.py`.

These tests are not currently connected to a Buildkite job. The single-request
case was run manually on one H800 with Qwen3-VL-8B-Instruct and a vLLM build based
on `e4a5b5537344b103ed10de7764378ef09604ef3d` with the EC tiering changes applied:
`1 passed`. The concurrent case requires a separate accelerator run before
claiming validation.
