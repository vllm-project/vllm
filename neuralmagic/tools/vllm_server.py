import logging
import shlex
import sys
import time
from subprocess import STDOUT, Popen
from tempfile import TemporaryFile
from typing import Dict, List
from urllib.parse import urljoin

import requests


def log_banner(logger: logging.Logger,
               label: str,
               body: str,
               level: int = logging.INFO):
    """
    Log a message in the "banner"-style format.
    :param logger: Instance of "logging.Logger" to use
    :param label: Label for the top of the banner
    :param body: Body content inside the banner
    :param level: Logging level to use (default: INFO)
    """

    banner = f"==== {label} ====\n{body}\n===="
    logger.log(level, "\n%s", banner)


class BaseVllmServerError(Exception):
    """Base exception class for VllmServer errors."""


class VllmServerDiedError(BaseVllmServerError):
    """Error indicating that the vLLM server died unexpectedly."""


class VllmServerTimeoutError(BaseVllmServerError):
    """Error indicating that the vLLM server took too long to become ready."""


class VllmServer:
    """Context manager for the lifecycle of a vLLM server"""

    def __init__(self, args: Dict[str, str], logger: logging.Logger) -> None:
        """Initialize a vLLM server
        :param args: dict of commands to pass to the server shell command
        :param logger: logging.Logger instance to use for logging
        """
        self._args = self._args_to_list(args)
        self._cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            *self._args,
        ]
        # TODO: Is there an alternative/short-flag to also check, e.g. "-p"?
        self._port = args.get("--port", "8000")
        self._logger = logger
        pass

    def __enter__(self):
        """Executes the server process and waits for it to become ready."""
        log_banner(self._logger, "server startup command",
                   shlex.join(self._cmd), logging.DEBUG)
        command = self._cmd
        self._output_file = TemporaryFile()
        self._process = Popen(command,
                              stdout=self._output_file.fileno(),
                              stderr=STDOUT)
        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Stops the server if it's still running and captures/logs its
        output"""
        if hasattr(self, "_process") and self._process.poll() is None:
            self._process.terminate()

        # capture and log server output
        self._output_file.seek(0)
        self.output = self._output_file.read().decode(errors="backslash")
        self._output_file.close()
        log_banner(self._logger, "server output", self.output, logging.DEBUG)

    def _args_to_list(self, args: Dict[str, str]) -> List[str]:
        """Convert a dict mapping of CLI args to a list
        :param args: `dict` containing CLI flags and their values (all strings)
        :return: flattened list to pass to a CLI
        """

        arg_list: List[str] = []
        for flag, value in args.items():
            # some error-checking safety as resulting list must be all strings
            if not isinstance(flag, str):
                error = f"all flags must be strings, got {type(flag)} ({flag})"
                raise ValueError(error)
            if not isinstance(value, str):
                error = (f"all values must be strings, got {type(value)} "
                         f"({value})")
                raise ValueError(error)

            arg_list.append(flag)
            if value != "":
                arg_list.append(value)

        return arg_list

    def _is_ready(self) -> bool:
        """Determine if server is ready based on response from `/health`
        endpoint
        :raises VllmServerDiedError: raised if the server process is no longer
            running
        :return: boolean state of the server's readiness
        """
        health_url = urljoin(f"http://localhost:{self._port}", "/health")
        try:
            rsp = requests.get(health_url, timeout=5)
            rsp.raise_for_status()
            return rsp.status_code == requests.status_codes.codes.ok
        except Exception as e:
            if self._process.poll() is not None:
                error = "server process not running when performing ready-check"
                raise VllmServerDiedError(error) from e
            return False

    def _wait_for_ready(self, limit: int = 600):
        """Wait for the server to become ready up to `limit` seconds
        (default: 600)
        :param limit: time in seconds to wait for the server to become ready
            (default: 600)
        :raises VllmServerTimeoutError: raised if the server does not become
            ready in the specified time limit
        """
        start = time.time()
        while True:
            if self._is_ready():
                return
            time.sleep(0.25)
            if time.time() - start > limit:
                error = f"did not become ready within {limit} seconds"
                raise VllmServerTimeoutError(error)
