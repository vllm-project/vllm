# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Callable
from ssl import SSLContext

from watchfiles import Change, awatch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SSLCertRefresher:
    """A class that monitors SSL certificate files and
    reloads them when they change.
    """

    def __init__(
        self,
        ssl_context: SSLContext,
        key_path: str | None = None,
        cert_path: str | None = None,
        ca_path: str | None = None,
    ) -> None:
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
                self._watch_files(
                    [self.key_path, self.cert_path], update_ssl_cert_chain
                )
            )

        # Setup CA files watcher
        def update_ssl_ca(change: Change, file_path: str) -> None:
            logger.info("Reloading SSL CA certificates")
            assert self.ca_path
            self.ssl.load_verify_locations(self.ca_path)

        self.watch_ssl_ca_task = None
        if self.ca_path:
            self.watch_ssl_ca_task = asyncio.create_task(
                self._watch_files([self.ca_path], update_ssl_ca)
            )

    async def _watch_files(self, paths, fun: Callable[[Change, str], None]) -> None:
        """Watch multiple file paths asynchronously."""
        logger.info("SSLCertRefresher monitors files: %s", paths)
        async for changes in awatch(*paths):
            try:
                for change, file_path in changes:
                    logger.info("File change detected: %s - %s", change.name, file_path)
                    fun(change, file_path)
            except Exception as e:
                logger.error(
                    "SSLCertRefresher failed taking action on file change. Error: %s", e
                )

    def stop(self) -> None:
        """Stop watching files."""
        if self.watch_ssl_cert_task:
            self.watch_ssl_cert_task.cancel()
            self.watch_ssl_cert_task = None
        if self.watch_ssl_ca_task:
            self.watch_ssl_ca_task.cancel()
            self.watch_ssl_ca_task = None
