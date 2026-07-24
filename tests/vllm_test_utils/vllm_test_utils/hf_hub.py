# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import shlex
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled

HF_HUB_RETRY_EXIT_STATUS = 75

_PLUGIN_MODULE = "vllm_test_utils.hf_hub"
_PLUGIN_NAME = "vllm-hf-hub-request-monitor"
_REQUEST_LOG_OPTION = "--hf-request-log"
_REQUEST_NODEID_EXTENSION = "vllm.hf_request_nodeid"
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_SESSION = "<session>"
_TRANSIENT_HTTPX_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
)


def _exception_chain(error: BaseException) -> Generator[BaseException, None, None]:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        grouped = getattr(current, "exceptions", ())
        if isinstance(grouped, tuple):
            pending.extend(grouped)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _request_from(error: BaseException) -> httpx.Request | None:
    for owner in (error, getattr(error, "response", None)):
        try:
            request = getattr(owner, "request", None)
        except (AttributeError, RuntimeError):
            continue
        if isinstance(request, httpx.Request):
            return request
    return None


def _retryable_request_nodeid(error: BaseException) -> str | None:
    for current in _exception_chain(error):
        response = getattr(current, "response", None)
        retryable = isinstance(current, _TRANSIENT_HTTPX_ERRORS) or (
            getattr(response, "status_code", None) in _RETRYABLE_STATUS_CODES
        )
        request = _request_from(current)
        if retryable and request is not None:
            nodeid = request.extensions.get(_REQUEST_NODEID_EXTENSION)
            if nodeid:
                return nodeid
    return None


def _is_offline_cache_miss(error: BaseException) -> bool:
    return any(
        isinstance(current, (LocalEntryNotFoundError, OfflineModeIsEnabled))
        for current in _exception_chain(error)
    )


def _safe_nodeid(nodeid: str) -> str:
    return nodeid.replace("\r", r"\r").replace("\n", r"\n").replace("\t", r"\t")


def _append_log(path: Path, rows: str) -> None:
    if not rows:
        return
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, rows.encode())
    finally:
        os.close(descriptor)


def _append_counts(path: Path, counts: Mapping[str, int]) -> None:
    _append_log(
        path,
        "".join(f"{count}\t{nodeid}\n" for nodeid, count in counts.items() if count),
    )


def _read_log(path: Path) -> tuple[Counter[str], bool]:
    counts: Counter[str] = Counter()
    retryable = False
    with path.open() as requests:
        for line in requests:
            if line == "retry\n":
                retryable = True
                continue
            count, separator, nodeid = line.rstrip("\n").partition("\t")
            if separator and count.isdecimal():
                counts[nodeid] += int(count)
    return counts, retryable


def _format_table(counts: Mapping[str, int]) -> str:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    lines = [f"{'REQUESTS':>8}  TEST"]
    lines.extend(f"{count:>8}  {nodeid}" for nodeid, count in rows)
    lines.append(f"{sum(counts.values()):>8}  TOTAL")
    return "\n".join(lines)


class HFHubRequestMonitor:
    """Count and attribute requests made through Hugging Face Hub clients."""

    def __init__(self, request_log: Path | None = None) -> None:
        self._request_log = request_log
        self._counts: Counter[str] = Counter()
        self._current_nodeid = _SESSION
        self._retryable_failure = False
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._factories: tuple[Callable[[], Any], Callable[[], Any]] | None = None

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._lock = threading.Lock()
            self._counts.clear()
            self._retryable_failure = False

    def _record_request(self, request: httpx.Request) -> None:
        self._reset_after_fork()
        with self._lock:
            nodeid = self._current_nodeid
            self._counts[nodeid] += 1
        request.extensions[_REQUEST_NODEID_EXTENSION] = nodeid

    async def _record_async_request(self, request: httpx.Request) -> None:
        self._record_request(request)

    def _wrap_factory(
        self, factory: Callable[[], Any], *, asynchronous: bool
    ) -> Callable[[], Any]:
        request_hook = (
            self._record_async_request if asynchronous else self._record_request
        )

        def monitored_client() -> Any:
            client = factory()
            hooks = client.event_hooks.setdefault("request", [])
            if request_hook not in hooks:
                hooks.insert(0, request_hook)
            return client

        return monitored_client

    def install(self) -> None:
        """Install listeners on the Hub's sync and async client factories."""
        if self._factories is not None:
            return
        from huggingface_hub.utils import _http as hf_http

        self._factories = (
            hf_http._GLOBAL_CLIENT_FACTORY,
            hf_http._GLOBAL_ASYNC_CLIENT_FACTORY,
        )
        hf_http.set_client_factory(
            self._wrap_factory(self._factories[0], asynchronous=False)
        )
        hf_http.set_async_client_factory(
            self._wrap_factory(self._factories[1], asynchronous=True)
        )

    def uninstall(self) -> None:
        """Restore the Hub client factories that preceded this monitor."""
        if self._factories is None:
            return
        from huggingface_hub.utils import _http as hf_http

        hf_http.set_client_factory(self._factories[0])
        hf_http.set_async_client_factory(self._factories[1])
        self._factories = None

    @property
    def counts(self) -> Counter[str]:
        """Return a snapshot of request counts in the current process."""
        self._reset_after_fork()
        with self._lock:
            return self._counts.copy()

    def _set_current_nodeid(self, nodeid: str) -> None:
        self._reset_after_fork()
        with self._lock:
            self._current_nodeid = _safe_nodeid(nodeid)

    def _flush(self, nodeid: str | None = None) -> None:
        if self._request_log is None:
            return
        self._reset_after_fork()
        with self._lock:
            if nodeid is None:
                counts = self._counts.copy()
                self._counts.clear()
            else:
                count = self._counts.pop(nodeid, 0)
                counts = Counter({nodeid: count})
        _append_counts(self._request_log, counts)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(
        self, item: Any, nextitem: Any
    ) -> Generator[None, Any, None]:
        self._set_current_nodeid(item.nodeid)
        try:
            yield
        finally:
            self._set_current_nodeid(_SESSION)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(
        self, item: Any, call: Any
    ) -> Generator[None, Any, None]:
        outcome = yield
        report = outcome.get_result()
        error = call.excinfo.value if call.excinfo is not None else None
        retryable = False
        if report.failed and error is not None:
            if hf_constants.is_offline_mode():
                retryable = _is_offline_cache_miss(error)
            else:
                retryable = _retryable_request_nodeid(error) == report.nodeid
        report.vllm_hf_retryable = retryable
        self._reset_after_fork()
        with self._lock:
            self._retryable_failure |= retryable
        if report.when == "teardown":
            self._flush(report.nodeid)

    def pytest_runtest_logreport(self, report: Any) -> None:
        """Receive retry attribution from pytest-forked children."""
        if getattr(report, "vllm_hf_retryable", False):
            self._reset_after_fork()
            with self._lock:
                self._retryable_failure = True

    def pytest_exception_interact(self, node: Any, call: Any, report: Any) -> None:
        """Handle transient errors raised during test collection."""
        if getattr(report, "when", None) != "collect" or call.excinfo is None:
            return
        error = call.excinfo.value
        retryable = (
            _is_offline_cache_miss(error)
            if hf_constants.is_offline_mode()
            else _retryable_request_nodeid(error) == _SESSION
        )
        if retryable:
            self._reset_after_fork()
            with self._lock:
                self._retryable_failure = True

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        self._flush()
        self._reset_after_fork()
        with self._lock:
            retryable = self._retryable_failure
        if exitstatus not in {0, 5} and retryable:
            session.exitstatus = HF_HUB_RETRY_EXIT_STATUS

    def pytest_unconfigure(self) -> None:
        """Flush pending counts and restore the original Hub clients."""
        self._flush()
        self.uninstall()


@dataclass(frozen=True)
class HFHubTestGroup:
    """Run a shell test group with HF request monitoring enabled."""

    command: str
    offline: bool = False
    pipefail: bool = False

    def _environment(self, request_log: Path) -> dict[str, str]:
        environment = os.environ.copy()
        pytest_options = shlex.split(environment.get("PYTEST_ADDOPTS", ""))
        pytest_options.extend(
            ["-p", _PLUGIN_MODULE, f"{_REQUEST_LOG_OPTION}={request_log}"]
        )
        environment["PYTEST_ADDOPTS"] = shlex.join(pytest_options)
        offline = "1" if self.offline else "0"
        environment.update(
            {
                "HF_HUB_OFFLINE": offline,
                "HF_DATASETS_OFFLINE": offline,
                "TRANSFORMERS_OFFLINE": offline,
            }
        )
        environment.pop("VLLM_CI_HF_HUB_MODE", None)
        return environment

    def run(self) -> int:
        """Run the group, print its request table, and return its exit status."""
        shell_command = ["/bin/bash"]
        if self.pipefail:
            shell_command.extend(["-o", "pipefail"])
        shell_command.extend(["-c", self.command])
        with tempfile.NamedTemporaryFile(prefix="vllm-hf-requests-") as request_log:
            path = Path(request_log.name)
            completed = subprocess.run(
                shell_command,
                env=self._environment(path),
                check=False,
            )
            counts, retryable = _read_log(path)
        print("Hugging Face Hub HTTP requests")
        print(_format_table(counts))
        if completed.returncode not in {0, 5} and retryable:
            return HF_HUB_RETRY_EXIT_STATUS
        return completed.returncode


def pytest_addoption(parser: Any) -> None:
    """Add the private aggregation path used by the test-group runner."""
    group = parser.getgroup("Hugging Face Hub monitoring")
    group.addoption(
        _REQUEST_LOG_OPTION,
        type=Path,
        help="append per-test Hugging Face Hub request counts to this file",
    )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_load_initial_conftests(
    early_config: Any,
) -> Generator[None, Any, None]:
    """Install the request listener before repository conftest imports."""
    if early_config.pluginmanager.hasplugin(_PLUGIN_NAME):
        yield
        return
    request_log = getattr(early_config.known_args_namespace, "hf_request_log", None)
    monitor = HFHubRequestMonitor(request_log)
    monitor.install()
    early_config.pluginmanager.register(monitor, _PLUGIN_NAME)
    outcome = yield
    if outcome.excinfo is None:
        return
    error = getattr(outcome.excinfo[1], "cause", outcome.excinfo[1])
    retryable = (
        _is_offline_cache_miss(error)
        if hf_constants.is_offline_mode()
        else _retryable_request_nodeid(error) == _SESSION
    )
    if retryable:
        monitor._flush()
        monitor.uninstall()
        if request_log is not None:
            _append_log(request_log, "retry\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run an HF-monitored test group from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="disable network access for Hub, Transformers, and Datasets clients",
    )
    parser.add_argument(
        "--pipefail",
        action="store_true",
        help="propagate failures from commands in a shell pipeline",
    )
    parser.add_argument("command", help="shell command for the test group")
    arguments = parser.parse_args(argv)
    return HFHubTestGroup(
        arguments.command,
        offline=arguments.offline,
        pipefail=arguments.pipefail,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
