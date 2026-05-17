# Windows Compatibility Work

This fork tracks native Windows fixes needed by OmniChat's embedded vLLM experiments.

## Current Branch

Branch: `windows-compat`

## Change

- `vllm.utils.network_utils.get_open_zmq_ipc_path()` now returns a loopback TCP ZMQ endpoint on Windows, because pyzmq does not support `ipc://` transport on Windows.
- Unix behavior is unchanged: non-Windows platforms still return `ipc://...`.
- Added a targeted unit test that simulates `sys.platform == "win32"` and verifies the returned ZMQ path is TCP.

## Validation

- `python -m py_compile vllm/utils/network_utils.py tests/utils_/test_network_utils.py` passes in the local vLLM spike environment.
- Running the upstream pytest target from the unbuilt source tree is blocked by missing compiled extension `vllm._C`; this branch still needs a full source build / wheel CI pass.

## Runtime Evidence

This mirrors the monkeypatch that allowed native Windows vLLM-Omni stage handshakes to progress from `Protocol not supported` to successful three-stage `Qwen/Qwen2.5-Omni-3B` text and audio output.
