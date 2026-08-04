# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Optional
from urllib.parse import quote as url_quote
import asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # Third Party
    from azure.storage.blob.aio import ContainerClient

logger = init_logger(__name__)


class AzureConnector(RemoteConnector):
    """
    Native Azure Blob Storage remote connector.

    Mirrors the S3Connector contract but talks to Azure Blob Storage via
    ``azure-storage-blob`` (async client). The SDK is imported lazily inside
    ``__init__`` so that ``azure-storage-blob`` remains an optional dependency.

    Keys are flattened into a single blob name per chunk (slashes replaced and
    URL-encoded), matching the flat layout S3Connector uses.
    """

    def __init__(
        self,
        container: str,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        account_url: Optional[str] = None,
        connection_string: Optional[str] = None,
        account_key: Optional[str] = None,
        sas_token: Optional[str] = None,
    ) -> None:
        """Initialize the Azure Blob connector.

        Exactly one credential path is used, resolved most-explicit-first:
        ``connection_string`` wins; otherwise ``account_url`` is required and
        is paired with ``account_key``, then ``sas_token``, then (if neither is
        given) ``DefaultAzureCredential``.

        Args:
            container: The target blob container name.
            loop: The asyncio event loop the connector runs its I/O on.
            local_cpu_backend: Backend used to allocate MemoryObj buffers.
            account_url: Storage account URL, e.g.
                ``https://<account>.blob.core.windows.net``. Required unless
                ``connection_string`` is provided.
            connection_string: Full Azure Storage connection string.
            account_key: Storage account shared key.
            sas_token: Shared Access Signature token.

        Raises:
            ImportError: If ``azure-storage-blob`` is not installed, or if
                ``azure-identity`` is not installed when falling back to
                ``DefaultAzureCredential``.
            ValueError: If neither ``connection_string`` nor ``account_url``
                is provided.
        """
        # initialize base class, which includes some common attributes
        super().__init__(local_cpu_backend.config, local_cpu_backend.metadata)

        # Lazy import keeps azure-storage-blob optional.
        try:
            # Third Party
            from azure.storage.blob.aio import BlobServiceClient
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required for the Azure connector. "
                "Install it with `pip install azure-storage-blob`."
            ) from e

        self.container = container
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.part_size = self.full_chunk_size_bytes

        # ---- Auth resolution (most explicit first) ----------------------
        if connection_string:
            logger.info("AzureConnector: authenticating via connection string")
            self.service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
        else:
            if account_url is None:
                raise ValueError(
                    "AzureConnector requires either `connection_string` or "
                    "`account_url` (e.g. https://<account>.blob.core.windows.net)."
                )
            if account_key:
                logger.info("AzureConnector: authenticating via account key")
                credential: object = account_key
            elif sas_token:
                logger.info("AzureConnector: authenticating via SAS token")
                credential = sas_token
            else:
                logger.info(
                    "AzureConnector: no explicit credential, "
                    "falling back to DefaultAzureCredential"
                )
                try:
                    # Third Party
                    from azure.identity.aio import DefaultAzureCredential
                except ImportError as e:
                    raise ImportError(
                        "azure-identity is required to use "
                        "DefaultAzureCredential. Install it with "
                        "`pip install azure-identity`, or provide an "
                        "explicit credential (connection_string, account_key, "
                        "or sas_token)."
                    ) from e

                credential = DefaultAzureCredential()
            self.service_client = BlobServiceClient(
                account_url=account_url, credential=credential
            )

        self.container_client: "ContainerClient" = (
            self.service_client.get_container_client(container)
        )

        # Cache of known blob sizes (read-only assumption, like S3Connector).
        self.object_size_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _blob_name(self, key_str: str) -> str:
        """Flatten a key into a safe blob name."""
        return url_quote(key_str.replace("/", "_"), safe="")

    async def _get_blob_size_async(self, key_str: str) -> int:
        """Return the blob size in bytes for ``key_str``, or 0 if the blob is
        absent or its properties cannot be read.
        """
        # Third Party
        from azure.core.exceptions import ResourceNotFoundError

        blob = self.container_client.get_blob_client(self._blob_name(key_str))
        try:
            props = await blob.get_blob_properties()
            return props.size or 0
        except ResourceNotFoundError:
            logger.debug("Blob not found: %s", key_str)
            return 0
        except Exception as e:
            logger.warning("Azure HEAD/properties error for %s: %s", key_str, e)
            return 0

    # ------------------------------------------------------------------ #
    # exists
    # ------------------------------------------------------------------ #
    async def exists(self, key: CacheEngineKey) -> bool:
        """Check whether a chunk for ``key`` exists in the container.

        Consults the local blob-size cache first to avoid a network round-trip
        when the size is already known.

        Args:
            key: The cache engine key to look up.

        Returns:
            True if a blob for the key exists, False otherwise.
        """
        key_str = key.to_string()
        cached = self.object_size_cache.get(key_str)
        if cached is not None:
            return cached > 0
        size = await self._get_blob_size_async(key_str)
        # Cache misses too (size 0) so repeated lookups of an absent key
        # don't trigger a network round-trip each time, matching get().
        self.object_size_cache[key_str] = size
        return size > 0

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronous variant of :meth:`exists`.

        Drives the async :meth:`exists` coroutine to completion on the
        connector's event loop, since the aio client exposes no sync API.

        Args:
            key: The cache engine key to look up.

        Returns:
            True if a blob for the key exists, False otherwise.
        """
        # The async aio client has no sync path; drive the coroutine on the loop.
        return asyncio.run_coroutine_threadsafe(self.exists(key), self.loop).result()

    # ------------------------------------------------------------------ #
    # get
    # ------------------------------------------------------------------ #
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Download the chunk for ``key`` into a freshly allocated MemoryObj.

        The blob is streamed directly into the preallocated buffer (zero-copy).

        Args:
            key: The cache engine key to fetch.

        Returns:
            The populated MemoryObj, or None on a cache miss, allocation
            failure, a size mismatch against the current model/chunk
            configuration, or any download error.
        """
        key_str = key.to_string()

        obj_size = self.object_size_cache.get(key_str)
        if obj_size is None:
            obj_size = await self._get_blob_size_async(key_str)
            if obj_size <= 0:
                self.object_size_cache[key_str] = 0
                return None
            self.object_size_cache[key_str] = obj_size

        memory_obj = self.local_cpu_backend.allocate(
            self.meta_shapes,
            self.meta_dtypes,
            self.meta_fmt,
        )
        if memory_obj is None:
            return None

        if obj_size != memory_obj.get_size():
            logger.error(
                "Size mismatch for %s: Azure has %s bytes, "
                "but current config expects %s bytes. "
                "This usually means the data was stored with a different "
                "chunk_size or model configuration.",
                key_str,
                obj_size,
                memory_obj.get_size(),
            )
            memory_obj.ref_count_down()
            return None

        try:
            blob = self.container_client.get_blob_client(self._blob_name(key_str))
            downloader = await blob.download_blob()
            # Zero-copy: stream straight into the preallocated buffer.
            await downloader.readinto(memory_obj.byte_array)
            return memory_obj
        except Exception as e:
            logger.error("Failed to download %s from Azure: %s", key_str, e)
            memory_obj.ref_count_down()
            return None

    # ------------------------------------------------------------------ #
    # put
    # ------------------------------------------------------------------ #
    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Upload ``memory_obj`` to the container under ``key``.

        Only full chunks are supported: a chunk whose physical size does not
        match the connector's expected part size is skipped with an error log.
        Existing blobs with the same name are overwritten.

        Note:
            The caller (``InstrumentedRemoteConnector.put``) owns the reference
            count of ``memory_obj`` and decrements it after this returns; this
            method must not call ``ref_count_down`` itself, matching
            ``S3Connector``.

        Args:
            key: The cache engine key to store under.
            memory_obj: The memory object holding the chunk bytes.
        """
        key_str = key.to_string()

        if memory_obj.get_physical_size() != self.part_size:
            logger.error(
                "Cannot upload %s: chunk size "
                "%s bytes does not match expected "
                "part size %s bytes. "
                "Partial/unfull chunks are not supported.",
                key_str,
                memory_obj.get_physical_size(),
                self.part_size,
            )
            return

        try:
            blob = self.container_client.get_blob_client(self._blob_name(key_str))
            # Pass the memoryview directly (zero-copy); bytes() would copy the
            # whole chunk.
            await blob.upload_blob(
                memory_obj.byte_array,
                overwrite=True,
                length=memory_obj.get_physical_size(),
            )
            self.object_size_cache[key_str] = memory_obj.get_physical_size()
            logger.debug("Uploaded %s to Azure successfully", key_str)
        except Exception as e:
            logger.error("Failed to upload %s to Azure: %s", key_str, e)

    # ------------------------------------------------------------------ #
    # list / close
    # ------------------------------------------------------------------ #
    async def list(self) -> List[str]:
        """List the names of all blobs currently in the container.

        Returns:
            A list of blob names.
        """
        names: List[str] = []
        async for blob in self.container_client.list_blobs():
            names.append(blob.name)
        return names

    async def close(self) -> None:
        """Close the underlying Azure Blob service client and release its
        network resources.
        """
        await self.service_client.close()
