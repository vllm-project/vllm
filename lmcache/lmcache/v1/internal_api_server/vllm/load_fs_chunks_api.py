# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List, Optional
import asyncio
import json

# Third Party
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class LoadFSChunksRequest(BaseModel):
    """Request model for loading FS chunks."""

    config_path: str
    max_chunks: Optional[int] = None
    max_failed_keys: int = 10


class LoadFSChunksResponse(BaseModel):
    """Response model for load-fs-chunks endpoint."""

    status: str
    loaded_chunks: int
    total_files: int
    failed_keys: List[str]
    config_path: str


class ErrorResponse(BaseModel):
    """Error response model for load-fs-chunks endpoint."""

    error: str
    message: str
    config_path: Optional[str] = None


router = APIRouter()
logger = init_logger(__name__)


@router.post(
    "/cache/load-fs-chunks",
    summary="Load chunks from FSConnector into hot cache",
    description="""
    Load chunk files from FSConnector storage into LocalCPUBackend's hot cache.
    """,
    responses={
        200: {
            "model": LoadFSChunksResponse,
            "description": "Chunks loaded successfully",
        },
        400: {"model": ErrorResponse, "description": "Invalid configuration file"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "LMCache engine not configured"},
    },
    tags=["cache-management"],
)
async def load_fs_chunks(
    request: Request,
    request_body: LoadFSChunksRequest,
):
    """
    Load chunk files from FSConnector into LocalCPUBackend hot cache.

    This endpoint loads all chunk files from the specified FSConnector directory
    into the LocalCPUBackend's hot cache by:
    1. Loading configuration from the specified config file
    2. Initializing RemoteBackend with FSConnector
    3. Listing all chunk files in the FSConnector directory
    4. Constructing CacheEngineKey from filenames
    5. Loading MemoryObj from files and putting into hot cache

    Args:
        request: The FastAPI request object containing application state
        request_body: Request body containing:
            - config_path: Path to LMCache engine configuration file
            - max_chunks: Optional limit on number of chunks to load
            - max_failed_keys: Maximum failed keys to report (default: 10)

    Returns:
        PlainTextResponse: JSON response with loading statistics

    Raises:
        HTTPException: Various error conditions with appropriate status codes

    Example Request:
        ```bash
        curl -X POST "http://localhost:8000/cache/load-fs-chunks" \
             -H "Content-Type: application/json" \
             -d '{
                   "config_path": "/path/to/lmcache.yaml", 
                   "max_chunks": 100,
                   "max_failed_keys": 10
                 }'
        ```

    Example Response (Success):
        ```json
        {
          "status": "success",
          "loaded_chunks": 95,
          "total_files": 100,
          "failed_keys": ["key1", "key2"],
          "config_path": "/path/to/lmcache.yaml"
        }
        ```

    Example Response (Error):
        ```json
        {
          "error": "Failed to load chunks from FSConnector",
          "message": "Configuration file not found",
          "config_path": "/path/to/lmcache.yaml"
        }
        ```
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if not lmcache_engine:
        error_info = {
            "error": "/cache/load-fs-chunks API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )

    remote_backend = None
    try:
        config = await _load_config_from_file(request_body.config_path)
        local_cpu_backend = lmcache_engine.storage_manager.allocator_backend

        remote_backend = await _initialize_remote_backend(
            config,
            lmcache_engine.metadata,
            local_cpu_backend,
            lmcache_engine.storage_manager.loop,
        )

        result = await _load_chunks_from_fs_connector(
            remote_backend,
            local_cpu_backend,
            request_body.max_chunks,
            request_body.max_failed_keys,
        )

        success_info = {
            "status": "success",
            "loaded_chunks": result["loaded_chunks"],
            "total_files": result["total_files"],
            "failed_keys": result["failed_keys"],
            "config_path": request_body.config_path,
        }

        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error("Unexpected error in load_fs_chunks: %s", e, exc_info=True)
        error_info = {
            "error": "Failed to load chunks from FSConnector",
            "message": str(e),
            "config_path": request_body.config_path,
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
    finally:
        if remote_backend is not None:
            remote_backend.close()


async def _load_config_from_file(config_path: str) -> LMCacheEngineConfig:
    """Load configuration from yaml file."""
    try:
        return LMCacheEngineConfig.from_file(config_path)
    except Exception as e:
        logger.error("Failed to load config from %s: %s", config_path, e)
        raise HTTPException(
            status_code=400, detail="Invalid configuration file: %s" % str(e)
        ) from e


async def _initialize_remote_backend(
    config: LMCacheEngineConfig, metadata, local_cpu_backend, loop
) -> RemoteBackend:
    """Initialize RemoteBackend with FSConnector."""
    try:
        remote_backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        remote_backend.init_connection()
        return remote_backend
    except Exception as e:
        logger.error("Failed to initialize RemoteBackend: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to initialize RemoteBackend: %s" % str(e)
        ) from e


async def _load_chunks_from_fs_connector(
    remote_backend: RemoteBackend,
    local_cpu_backend,
    max_chunks: Optional[int] = None,
    max_failed_keys: int = 10,
) -> Dict:
    """Load chunks from FSConnector into LocalCPUBackend."""
    connector = remote_backend.connection
    if not connector:
        raise HTTPException(status_code=500, detail="FSConnector not initialized")

    try:
        chunk_files = await connector.list()
        total_files = len(chunk_files)

        if max_chunks:
            chunk_files = chunk_files[:max_chunks]

        logger.info("Found %d chunk files to load", len(chunk_files))

        loaded_chunks = 0
        failed_keys: List[str] = []

        # Use semaphore to control concurrency (max 10 concurrent tasks)
        semaphore = asyncio.Semaphore(10)

        async def process_chunk(chunk_filename: str) -> bool:
            """
            Process a single chunk file from FSConnector.

            This function is called for each chunk file and performs:
            - Transforms filename to CacheEngineKey format
            - Loads MemoryObj data from remote backend
            - Places chunk into local CPU backend hot cache
            - Handles errors and tracks failed keys

            Args:
                chunk_filename: Name of the chunk file from FSConnector

            Returns:
                bool: True if chunk was successfully loaded, False otherwise

            Note:
                This function runs with semaphore control to limit concurrency
                and ensure system stability during bulk loading operations.
            """
            async with semaphore:
                key_str = chunk_filename.replace("-SEP-", "/")
                try:
                    key = CacheEngineKey.from_string(key_str)

                    # Get data from remote backend
                    memory_obj = await asyncio.get_event_loop().run_in_executor(
                        None, remote_backend.get_blocking, key
                    )

                    if memory_obj is None:
                        failed_keys.append(key_str)
                        logger.warning("Failed to load chunk: %s", key_str)
                        return False

                    # Put into local cpu backend and immediately release reference
                    local_cpu_backend.submit_put_task(key, memory_obj)
                    memory_obj.ref_count_down()
                    return True

                except Exception as e:
                    failed_keys.append(key_str)
                    logger.warning("Error processing chunk %s: %s", key_str, e)
                    return False

        # Process all chunks concurrently
        tasks = [process_chunk(chunk_filename) for chunk_filename in chunk_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful loads
        loaded_chunks = sum(1 for result in results if result is True)

        if loaded_chunks > 0 and loaded_chunks % 100 == 0:
            logger.info("Loaded %d chunks...", loaded_chunks)

        logger.info(
            "Successfully loaded %d chunks from %d files",
            loaded_chunks,
            total_files,
        )

        return {
            "loaded_chunks": loaded_chunks,
            "total_files": total_files,
            "failed_keys": failed_keys[:max_failed_keys],
        }

    except Exception as e:
        logger.error("Error in chunk loading process: %s", e)
        raise HTTPException(
            status_code=500, detail="Chunk loading failed: %s" % str(e)
        ) from e
