# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import ssl
from dataclasses import dataclass
from ssl import SSLContext
from typing import Any, Callable, Optional

import uvicorn
from watchfiles import Change, awatch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SSLConfig:
    """Shared SSL configuration for vLLM servers."""
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    ssl_cert_reqs: Optional[int] = None
    enable_ssl_refresh: bool = False

    @property
    def is_ssl_enabled(self) -> bool:
        """Check if SSL is enabled (requires both keyfile and certfile)."""
        return bool(self.ssl_keyfile and self.ssl_certfile)

    def validate(self) -> None:
        """Validate SSL configuration and file accessibility."""
        if not self.is_ssl_enabled:
            return

        # validate required files exist and are readable
        for file_path, name in [
            (self.ssl_keyfile, "SSL key file"),
            (self.ssl_certfile, "SSL certificate file"),
        ]:
            if file_path is None:
                raise ValueError(
                    f"{name} is required for SSL but not specified")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{name} not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"{name} not readable: {file_path}")

        # validate optional CA file if specified
        if self.ssl_ca_certs:
            if not os.path.exists(self.ssl_ca_certs):
                raise FileNotFoundError(
                    f"SSL CA file not found: {self.ssl_ca_certs}")
            if not os.access(self.ssl_ca_certs, os.R_OK):
                raise PermissionError(
                    f"SSL CA file not readable: {self.ssl_ca_certs}")

    def to_uvicorn_kwargs(self) -> dict[str, Any]:
        """Convert SSL config to uvicorn configuration kwargs."""
        if not self.is_ssl_enabled:
            return {}

        config: dict[str, Any] = {
            "ssl_keyfile": self.ssl_keyfile,
            "ssl_certfile": self.ssl_certfile,
        }

        if self.ssl_ca_certs is not None:
            config["ssl_ca_certs"] = self.ssl_ca_certs
        if self.ssl_cert_reqs is not None:
            config["ssl_cert_reqs"] = int(self.ssl_cert_reqs)

        return config

    def create_ssl_cert_refresher(
            self,
            uvicorn_config: uvicorn.Config) -> Optional['SSLCertRefresher']:
        """Create SSL certificate refresher if SSL refresh is enabled."""
        if not self.enable_ssl_refresh or not self.is_ssl_enabled:
            return None

        if uvicorn_config.ssl is None:
            raise ValueError("SSL context not available in uvicorn config")

        return SSLCertRefresher(ssl_context=uvicorn_config.ssl,
                                key_path=self.ssl_keyfile,
                                cert_path=self.ssl_certfile,
                                ca_path=self.ssl_ca_certs)

    @classmethod
    def from_args(cls, args) -> 'SSLConfig':
        """Create SSL config from parsed arguments."""
        return cls(
            ssl_keyfile=getattr(args, 'ssl_keyfile', None),
            ssl_certfile=getattr(args, 'ssl_certfile', None),
            ssl_ca_certs=getattr(args, 'ssl_ca_certs', None),
            ssl_cert_reqs=getattr(args, 'ssl_cert_reqs', None),
            enable_ssl_refresh=getattr(args, 'enable_ssl_refresh', False),
        )

    def create_ssl_context(self) -> 'ssl.SSLContext':
        """Create SSL context for client connections."""
        if not self.is_ssl_enabled:
            raise ValueError("SSL is not enabled")

        context = ssl.create_default_context()

        # Load CA certificates if provided
        if self.ssl_ca_certs:
            context.load_verify_locations(cafile=self.ssl_ca_certs)
            # Keep default verification settings when CA is provided
        else:
            # For self-signed certificates without CA, disable verification
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        # Apply certificate requirements if specified
        if (reqs := self.ssl_cert_reqs) is not None:
            context.verify_mode = ssl.VerifyMode(reqs)

        return context

    def get_protocol(self) -> str:
        """Get protocol string (http or https)."""
        return "https" if self.is_ssl_enabled else "http"

    def format_listen_address(self, host: str, port: int) -> str:
        """Format complete listen address with protocol."""
        return f"{self.get_protocol()}://{host}:{port}"


class SSLCertRefresher:
    """A class that monitors SSL certificate files and
    reloads them when they change.
    """

    def __init__(self,
                 ssl_context: SSLContext,
                 key_path: Optional[str] = None,
                 cert_path: Optional[str] = None,
                 ca_path: Optional[str] = None) -> None:
        self.ssl = ssl_context
        self.key_path = key_path
        self.cert_path = cert_path
        self.ca_path = ca_path

        # Setup certification chain watcher
        def update_ssl_cert_chain(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL certificate chain")
            assert self.key_path and self.cert_path
            self.ssl.load_cert_chain(self.cert_path, self.key_path)

        self.watch_ssl_cert_task = None
        if self.key_path and self.cert_path:
            self.watch_ssl_cert_task = asyncio.create_task(
                self._watch_files([self.key_path, self.cert_path],
                                  update_ssl_cert_chain))

        # Setup CA files watcher
        def update_ssl_ca(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL CA certificates")
            assert self.ca_path
            self.ssl.load_verify_locations(self.ca_path)

        self.watch_ssl_ca_task = None
        if self.ca_path:
            self.watch_ssl_ca_task = asyncio.create_task(
                self._watch_files([self.ca_path], update_ssl_ca))

    async def _watch_files(self, paths, fun: Callable[[Change, str],
                                                      None]) -> None:
        """Watch multiple file paths asynchronously."""
        logger.info("SSLCertRefresher monitors files: %s", paths)
        async for changes in awatch(*paths):
            try:
                for change, file_path in changes:
                    logger.info("File change detected: %s - %s", change.name,
                                file_path)
                    fun(change, file_path)
            except Exception as e:
                logger.error(
                    "SSLCertRefresher failed taking action on file change. "
                    "Error: %s", e)

    def stop(self) -> None:
        """Stop watching files."""
        if self.watch_ssl_cert_task:
            self.watch_ssl_cert_task.cancel()
            self.watch_ssl_cert_task = None
        if self.watch_ssl_ca_task:
            self.watch_ssl_ca_task.cancel()
            self.watch_ssl_ca_task = None
