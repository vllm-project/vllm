# SPDX-License-Identifier: Apache-2.0
"""TDD tests for T4.5 — Boot-time probe for spec-decode cross-rank issues.

Test contract:
1. Module imports cleanly
2. probe() function signature accepts url/model/api_key/timeout
3. Failure message includes #41190 marker when cudaErrorIllegalAddress in response
4. Failure message includes #40941 marker on workspace AssertionError
5. CLI requires --url and --model (returns exit 2 otherwise)
6. CLI exit codes: 0 success, 1 probe-failure, 2 invocation-error

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch



def test_boot_probe_imports():
    """Module imports cleanly without optional deps."""
    from vllm._genesis.utils import boot_probe
    assert hasattr(boot_probe, "probe")
    assert hasattr(boot_probe, "main")


def test_probe_signature():
    """probe() has expected signature."""
    import inspect
    from vllm._genesis.utils.boot_probe import probe
    sig = inspect.signature(probe)
    assert "url" in sig.parameters
    assert "model" in sig.parameters
    assert "api_key" in sig.parameters
    assert "timeout" in sig.parameters


def test_probe_detects_41190_marker_in_response():
    """When response contains cudaErrorIllegalAddress, marker [#41190-class] is added."""
    from vllm._genesis.utils.boot_probe import probe

    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "Internal Server Error: cudaErrorIllegalAddress at sync"

    with patch("requests.post", return_value=fake_response):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="test-model",
            api_key="key",
            timeout=10,
        )
    assert ok is False
    assert "#41190" in msg


def test_probe_detects_workspace_lock_marker():
    """When response contains workspace AssertionError, marker [#40941] added."""
    from vllm._genesis.utils.boot_probe import probe

    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "AssertionError in workspace_manager: locked but grow needed"

    with patch("requests.post", return_value=fake_response):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="m",
            api_key="k",
            timeout=10,
        )
    assert ok is False
    assert "#40941" in msg


def test_probe_detects_oom_marker():
    """OOM in response gets [OOM] marker for memory_pool diagnostics."""
    from vllm._genesis.utils.boot_probe import probe

    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "CUDA out of memory: tried to allocate"

    with patch("requests.post", return_value=fake_response):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="m",
            api_key="k",
            timeout=10,
        )
    assert ok is False
    assert "OOM" in msg


def test_probe_handles_timeout():
    """Timeout caught with diagnostic message."""
    import requests
    from vllm._genesis.utils.boot_probe import probe

    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="m",
            api_key="k",
            timeout=5,
        )
    assert ok is False
    assert "TIMEOUT" in msg
    assert "cold compile" in msg.lower() or "5" in msg


def test_probe_handles_connection_error():
    """Connection refused returns descriptive error."""
    import requests
    from vllm._genesis.utils.boot_probe import probe

    with patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection refused"),
    ):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="m",
            api_key="k",
            timeout=10,
        )
    assert ok is False
    assert "CONNECTION" in msg


def test_probe_success_returns_elapsed():
    """Successful probe returns elapsed time in message."""
    from vllm._genesis.utils.boot_probe import probe

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "hi there"}}]
    }

    with patch("requests.post", return_value=fake_response):
        ok, msg = probe(
            url="http://localhost:8000/v1/x",
            model="m",
            api_key="k",
            timeout=10,
        )
    assert ok is True
    assert "probe passed" in msg
    assert "ms" in msg


def test_main_exits_2_on_missing_args(capsys):
    """CLI returns exit 2 when --url or --model missing."""
    from vllm._genesis.utils.boot_probe import main

    # No env vars, no args
    with patch.dict("os.environ", {}, clear=False) as _:
        # Clear our env vars
        import os
        for k in ("GENESIS_BOOT_PROBE_URL", "GENESIS_BOOT_PROBE_MODEL"):
            os.environ.pop(k, None)
        rc = main(argv=[])
    assert rc == 2
    err = capsys.readouterr().err
    assert "url" in err.lower() and "model" in err.lower()


def test_main_exits_1_on_probe_failure(monkeypatch):
    """CLI returns exit 1 when probe fails."""
    from vllm._genesis.utils import boot_probe

    monkeypatch.setattr(
        boot_probe, "probe",
        lambda **kw: (False, "simulated failure"),
    )
    rc = boot_probe.main(argv=[
        "--url", "http://localhost:8000/v1/x",
        "--model", "m",
    ])
    assert rc == 1


def test_main_exits_0_on_probe_success(monkeypatch):
    """CLI returns exit 0 when probe succeeds."""
    from vllm._genesis.utils import boot_probe

    monkeypatch.setattr(
        boot_probe, "probe",
        lambda **kw: (True, "probe passed in 100ms"),
    )
    rc = boot_probe.main(argv=[
        "--url", "http://localhost:8000/v1/x",
        "--model", "m",
        "--quiet",
    ])
    assert rc == 0
