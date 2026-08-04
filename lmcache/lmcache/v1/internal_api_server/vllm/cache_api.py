# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, Any, Callable, List, Optional, Tuple
import asyncio
import hashlib
import json
import traceback

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    compress_slot_mapping,
    parse_mixed_slot_mapping,
)
from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

router = APIRouter()


def _parse_tokens_from_params(
    tokens_mock: Optional[str],
) -> Tuple[Optional[List[int]], Optional[dict]]:
    """Parse tokens from input parameters.

    Args:
        tokens_mock: Two comma-separated numbers specifying start and end of token range
            - Example: "0,100" generates tokens [0, 1, 2, ..., 99]
            - Example: "50,75" generates tokens [50, 51, 52, ..., 74]

    Returns:
        Tuple of (tokens list, error dict).
        If error dict is not None, tokens will be None.
    """
    # TODO(baoloongmao): Add support for tokens_input parameter to read tokens from file
    if tokens_mock:
        try:
            parts = tokens_mock.split(",")
            if len(parts) != 2:
                raise ValueError("tokens_mock must contain exactly 2 numbers")
            start, end = int(parts[0].strip()), int(parts[1].strip())
            if start >= end:
                raise ValueError("start must be less than end")
            tokens = list(range(start, end))
            return tokens, None
        except ValueError as e:
            return None, {
                "error": "Invalid tokens_mock format",
                "message": f"tokens_mock must be 'start,end': {str(e)}",
            }
    else:
        return None, {
            "error": "Missing parameters",
            "message": "Must specify either tokens_input or tokens_mock",
        }


def _create_error_response(error_info: dict, status_code: int) -> PlainTextResponse:
    """Create a standardized error response.

    Args:
        error_info: Dictionary containing error information
        status_code: HTTP status code

    Returns:
        PlainTextResponse with error information
    """
    return PlainTextResponse(
        content=json.dumps(error_info, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _check_lmcache_engine(
    request: Request,
) -> Tuple[Optional["LMCacheEngine"], Optional[PlainTextResponse]]:
    """Check if LMCache engine is available.

    Args:
        request: FastAPI request object

    Returns:
        Tuple of (lmcache_engine, error_response).
        If error_response is not None, engine will be None.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
    if not lmcache_engine:
        error_info = {
            "error": "LMCache API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return None, _create_error_response(error_info, 503)
    return lmcache_engine, None


def _get_kvcaches_and_device(engine):
    """Get kvcaches and device from engine's gpu_connector.

    Args:
        engine: LMCache engine instance

    Returns:
        Tuple of (kvcaches, device).
        kvcaches may be None if not available.
        device defaults to "cpu" if kvcaches not available.
    """
    kvcaches = None
    device = "cpu"  # Default device

    if engine.gpu_connector:
        kvcaches = engine.gpu_connector.kvcaches
        if kvcaches is not None and len(kvcaches) > 0:
            device = kvcaches[0].device
            logger.debug("Using kvcaches device: %s", device)
        else:
            logger.warning(
                "gpu_connector.kvcaches is None or empty. "
                "Make sure post_init was called with kvcaches."
            )

    return kvcaches, device


def _compute_tensor_checksum(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of a tensor."""
    # Move to CPU and convert to bytes for hashing
    # Handle BFloat16 which is not supported by numpy
    if tensor.dtype == torch.bfloat16:
        # Convert bfloat16 to float32 for numpy compatibility
        tensor = tensor.to(torch.float32)

    tensor_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


def _slice_by_slot_dim(
    kv_at_slots: torch.Tensor, start_idx: int, end_idx: int
) -> torch.Tensor:
    """Slice tensor by slot dimension based on tensor ndim."""
    if kv_at_slots.ndim == 4:
        # MHA: [2, num_slots, num_heads, head_size]
        return kv_at_slots[:, start_idx:end_idx, :, :]
    elif kv_at_slots.ndim == 3:
        # 4D format: [num_slots, num_heads, head_size]
        return kv_at_slots[start_idx:end_idx, :, :]
    else:
        # MLA: [num_slots, head_size]
        return kv_at_slots[start_idx:end_idx, :]


def _extract_kv_at_slots(
    kv_tensor: torch.Tensor, slot_tensor: torch.Tensor
) -> torch.Tensor:
    """Extract KV data at specified slot positions from kv_tensor.

    Handles different kv_tensor formats:
    - MHA (5D): [2, num_blocks, block_size, num_heads, head_size]
    - MLA (3D): [num_blocks, block_size, head_size]

    The slot_mapping is calculated as:
        slot_idx = block_id * block_size + block_offset

    This means we can reshape the tensor to flatten (num_blocks, block_size)
    into a single slot dimension and index directly.

    Args:
        kv_tensor: The KV cache tensor for a single layer.
        slot_tensor: Tensor of slot indices to extract.

    Returns:
        Tensor with KV data at the specified slots.
        - MHA: shape [2, num_slots, num_heads, head_size]
        - MLA: shape [num_slots, head_size]
    """
    ndim = kv_tensor.ndim

    if ndim == 5:
        # MHA format: [2, num_blocks, block_size, num_heads, head_size]
        # Reshape to [2, num_blocks * block_size, num_heads, head_size]
        # then index by slot_tensor on dimension 1
        kv_2d = 2
        num_heads = kv_tensor.shape[3]
        head_size = kv_tensor.shape[4]
        kv_reshaped = kv_tensor.reshape(kv_2d, -1, num_heads, head_size)
        return kv_reshaped[:, slot_tensor, :, :]
    elif ndim == 3:
        # MLA format: [num_blocks, block_size, head_size]
        # Reshape to [num_blocks * block_size, head_size]
        head_size = kv_tensor.shape[2]
        kv_reshaped = kv_tensor.reshape(-1, head_size)
        return kv_reshaped[slot_tensor, :]
    elif ndim == 4:
        # Alternative format: [num_blocks, block_size, num_heads, head_size]
        # (used in some test cases)
        num_heads = kv_tensor.shape[2]
        head_size = kv_tensor.shape[3]
        kv_reshaped = kv_tensor.reshape(-1, num_heads, head_size)
        return kv_reshaped[slot_tensor, :, :]
    else:
        # Fallback: try the original approach
        logger.warning(
            "Unknown kv_tensor ndim=%d, shape=%s. Using fallback indexing.",
            ndim,
            kv_tensor.shape,
        )
        return kv_tensor.view(-1, *kv_tensor.shape[2:])[slot_tensor]


def compute_kvcache_checksums(
    lmcache_adapter,
    slot_indices: list[int],
    chunk_size: Optional[int] = None,
    layerwise: bool = False,
) -> Optional[dict[str, Any]]:
    """Compute MD5 checksums for kvcaches at specified slot positions.

    This method is used by the kvcache check API to verify that stored
    and retrieved kvcaches are identical.

    The slot_mapping is calculated in vllm_v1_adapter.py as:
        slot_idx = block_id * block_size + block_offset

    For vLLM kv_cache formats:
    - MHA (5D): [2, num_blocks, block_size, num_heads, head_size]
    - MLA (3D): [num_blocks, block_size, head_size]

    Args:
        lmcache_adapter: The LMCache adapter containing kv_caches.
        slot_indices: List of slot indices to compute checksums for.
        chunk_size: Optional chunk size for computing per-chunk checksums.
            If provided, will compute checksums for each chunk.
        layerwise: If True, output per-layer checksums for each chunk.
            If False (default), output one checksum per chunk (all layers combined).

    Returns:
        Dictionary containing:
        - 'chunk_checksums': (if layerwise=True) dict mapping layer names to
          list of per-chunk checksums
        - 'chunk_checksums': (if layerwise=False) list of checksums, one per chunk
          (each checksum covers all layers for that chunk)
        Returns None if kv_caches is not available.
    """
    if not lmcache_adapter.kv_caches:
        logger.warning("kv_caches is empty, cannot compute checksums")
        return None

    if chunk_size is None or chunk_size <= 0:
        return {"chunk_checksums": [], "chunk_size": chunk_size, "num_chunks": 0}

    num_slots = len(slot_indices)
    num_chunks = (num_slots + chunk_size - 1) // chunk_size

    # Pre-extract all layer data at slot positions
    layer_data_at_slots: dict[str, torch.Tensor] = {}
    for layer_name, kv_tensor in lmcache_adapter.kv_caches.items():
        try:
            slot_tensor = torch.tensor(
                slot_indices, dtype=torch.long, device=kv_tensor.device
            )
            layer_data_at_slots[layer_name] = _extract_kv_at_slots(
                kv_tensor, slot_tensor
            )
        except Exception as e:
            logger.error("Failed to extract data for layer %s: %s", layer_name, str(e))
            layer_data_at_slots[layer_name] = None  # type: ignore

    if layerwise:
        # Output per-layer checksums for each chunk: {layer_name: [checksum1, ...]}
        chunk_checksums: dict[str, list[str]] = {}
        for layer_name, kv_at_slots in layer_data_at_slots.items():
            if kv_at_slots is None:
                chunk_checksums[layer_name] = ["error"] * num_chunks
                continue
            chunk_checksum_list: list[str] = []
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, num_slots)
                chunk_data = _slice_by_slot_dim(kv_at_slots, start_idx, end_idx)
                chunk_checksum_list.append(_compute_tensor_checksum(chunk_data))
            chunk_checksums[layer_name] = chunk_checksum_list
        return {
            "chunk_checksums": chunk_checksums,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
        }
    else:
        # Output one checksum per chunk (all layers combined): [checksum1, ...]
        chunk_checksums_list: list[str] = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_slots)
            md5_hash = hashlib.md5()
            for layer_name in sorted(layer_data_at_slots.keys()):
                kv_at_slots = layer_data_at_slots[layer_name]
                if kv_at_slots is None:
                    continue
                chunk_data = _slice_by_slot_dim(kv_at_slots, start_idx, end_idx)
                md5_hash.update(_compute_tensor_checksum(chunk_data).encode())
            chunk_checksums_list.append(md5_hash.hexdigest())
        return {
            "chunk_checksums": chunk_checksums_list,
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
        }


@router.delete("/cache/clear")
async def clear(
    request: Request,
    locations: Annotated[Optional[List[str]], Query()] = None,
    request_configs: Optional[dict] = None,
):
    """Clear cached data from the LMCache engine.

    This endpoint provides a way to clear cached KV (Key-Value) data from the
    LMCache engine. It can clear all cached data or selectively clear data
    from specific storage locations.

    Args:
        request (Request): The FastAPI request object containing application state.
        locations (Optional[List[str]], optional): List of storage backend locations
            to clear cache from. If None, clears from all available locations.
            Common values include ["LocalCPUBackend", "LocalDiskBackend"].
            Defaults to None.
        request_configs (Optional[dict], optional): Additional configuration
            parameters for the clear operation. Currently unused but reserved
            for future extensions. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response

    Example:
        Clear all cached data:
        ```bash
        curl -X DELETE "http://localhost:8000/cache/clear"
        # Response: {"status": "success", "num_removed": 10,
        #           "locations": null, "request_configs": null}
        ```

        Clear cache from specific locations:
        ```bash
        curl -X DELETE "http://localhost:8000/cache/clear?locations=LocalCPUBackend&locations=LocalDiskBackend"
        # Response: {"status": "success", "num_removed": 5,
        #           "locations": ["LocalCPUBackend", "LocalDiskBackend"],
        #           "request_configs": null}
        ```
    """
    try:
        lmcache_engine, error_response = _check_lmcache_engine(request)
        if error_response:
            return error_response

        assert lmcache_engine is not None
        num_removed = lmcache_engine.clear(  # type: ignore[attr-defined]
            locations=locations, request_configs=request_configs
        )
        success_info = {
            "status": "success",
            "num_removed": num_removed,
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_info = {"error": "Failed to clear cache", "message": str(e)}
        return _create_error_response(error_info, 500)


def _process_tokens_request(
    request: Request,
    tokens_mock: Optional[str],
) -> Tuple[Optional[object], Optional[List[int]], Optional[PlainTextResponse]]:
    """Process tokens request and validate parameters.

    Args:
        request: FastAPI request object
        tokens_mock: Mock token range specification

    Returns:
        Tuple of (lmcache_engine, tokens, error_response).
        If error_response is not None, the other values will be None.
    """
    lmcache_engine, error_response = _check_lmcache_engine(request)
    if error_response:
        return None, None, error_response

    tokens, error_info = _parse_tokens_from_params(tokens_mock)
    if error_info:
        status_code = 400 if error_info["error"] != "File not found" else 404
        return None, None, _create_error_response(error_info, status_code)

    return lmcache_engine, tokens, None


def _execute_cache_operation(
    operation_name: str,
    operation_func: Callable,
    lmcache_engine: object,
    tokens: List[int],
) -> PlainTextResponse:
    """Execute a cache operation and return standardized response.

    Args:
        operation_name: Name of the operation for error messages
        operation_func: Function to execute the operation
        lmcache_engine: LMCache engine instance
        tokens: List of token IDs

    Returns:
        PlainTextResponse with operation result
    """
    try:
        result = operation_func(lmcache_engine, tokens)
        success_info = {
            "status": "success",
            "num_tokens": len(tokens),
        }
        if result is not None:
            success_info.update(result)
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        # Log the full traceback for debugging
        tb_str = traceback.format_exc()
        logger.error("Failed to %s: %s\n%s", operation_name, e, tb_str)

        # Include more detailed error info in response
        error_message = str(e) if str(e) else f"Exception type: {type(e).__name__}"
        error_info = {
            "error": f"Failed to {operation_name}",
            "message": error_message,
            "exception_type": type(e).__name__,
        }
        return _create_error_response(error_info, 500)


@router.post("/cache/store")
async def store(
    request: Request,
    tokens_mock: Optional[str] = None,
):
    """Store KV cache data into the LMCache engine.

    This endpoint provides a way to store KV cache data by generating mock tokens.

    Args:
        request (Request): The FastAPI request object containing application state.
        tokens_mock (Optional[str], optional): Two comma-separated numbers specifying
            the start and end of a token range. Example: "0,100" generates tokens
            from 0 to 99. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response with operation status

    Example:
        Store with mock tokens:
        ```bash
        curl -X POST "http://localhost:8000/cache/store?tokens_mock=0,100"
        # Response: {"status": "success", "num_tokens": 100}
        ```
    """
    lmcache_engine, tokens, error_response = _process_tokens_request(
        request, tokens_mock
    )
    if error_response:
        return error_response

    assert tokens is not None
    assert lmcache_engine is not None

    def _store_operation(engine, token_list):
        # Get kvcaches and device using the shared function
        kvcaches, device = _get_kvcaches_and_device(engine)

        # Create slot mapping for the tokens
        slot_mapping = torch.arange(len(token_list), dtype=torch.long, device=device)

        logger.debug(
            "Storing %d tokens with slot_mapping on device %s",
            len(token_list),
            device,
        )

        engine.store(
            req_id="cache_api_store",
            tokens=token_list,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )
        return None

    return _execute_cache_operation(
        "store cache", _store_operation, lmcache_engine, tokens
    )


@router.post("/cache/retrieve")
async def retrieve(
    request: Request,
    tokens_mock: Optional[str] = None,
):
    """Retrieve KV cache data from the LMCache engine.

    This endpoint provides a way to retrieve KV cache data by generating mock tokens.

    Args:
        request (Request): The FastAPI request object containing application state.
        tokens_mock (Optional[str], optional): Two comma-separated numbers specifying
            the start and end of a token range. Example: "0,100" generates tokens
            from 0 to 99. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response with operation status

    Example:
        Retrieve with mock tokens:
        ```bash
        curl -X POST "http://localhost:8000/cache/retrieve?tokens_mock=0,100"
        # Response: {"status": "success", "num_tokens": 100, "num_retrieved": 80}
        ```
    """
    lmcache_engine, tokens, error_response = _process_tokens_request(
        request, tokens_mock
    )
    if error_response:
        return error_response

    assert tokens is not None
    assert lmcache_engine is not None

    def _retrieve_operation(engine, token_list):
        # Get kvcaches and device using the shared function
        kvcaches, device = _get_kvcaches_and_device(engine)

        # Create slot_mapping for retrieve operation
        slot_mapping = torch.arange(len(token_list), dtype=torch.long, device=device)

        logger.debug(
            "Retrieving %d tokens with slot_mapping on device %s",
            len(token_list),
            device,
        )

        ret_mask = engine.retrieve(
            req_id="cache_api_retrieve",
            tokens=token_list,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )
        num_retrieved = int(ret_mask.sum().item())
        return {"num_retrieved": num_retrieved}

    return _execute_cache_operation(
        "retrieve cache", _retrieve_operation, lmcache_engine, tokens
    )


@router.get("/cache/kvcache/check")
async def kvcache_check(
    request: Request,
    slot_mapping: Optional[str] = None,
    chunk_size: Optional[int] = None,
    layerwise: bool = False,
):
    """Compute checksum for kvcaches at specified slot_mapping positions.

    This endpoint is used to verify that stored and retrieved kvcaches are identical.

    Args:
        request (Request): The FastAPI request object containing application state.
        slot_mapping (Optional[str], optional): Slot indices in comma-separated format,
            supports single numbers and range expressions.
            Examples: "0,1,2,3", "1,2,3,[9,12],17,19". Defaults to None.
        chunk_size (Optional[int], optional): Chunk size for computing checksums.
            Each chunk contains `chunk_size` slots. Required parameter.
        layerwise (bool, optional): If True, output per-layer checksums for each chunk.
            If False (default), output one checksum per chunk (all layers combined).

    Returns:
        PlainTextResponse: A JSON response containing checksums.

    Example:
        ```bash
        # layerwise=false (default): one checksum per chunk (all layers combined)
        curl -X GET "http://localhost:8000/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2"
        # Response: {
        #   "status": "success",
        #   "slot_mapping_ranges": [[0, 3]],
        #   "chunk_size": 2,
        #   "num_chunks": 2,
        #   "chunk_checksums": ["checksum_chunk0", "checksum_chunk1"],
        #   "layerwise": false
        # }

        # layerwise=true: per-layer checksums for each chunk
        curl -X GET "http://localhost:8000/cache/kvcache/check?slot_mapping=0,1,2,3&chunk_size=2&layerwise=true"
        # Response: {
        #   "status": "success",
        #   "slot_mapping_ranges": [[0, 3]],
        #   "chunk_size": 2,
        #   "num_chunks": 2,
        #   "chunk_checksums": {
        #       "layer_0": ["checksum_chunk0", "checksum_chunk1"],
        #       "layer_1": ["checksum_chunk0", "checksum_chunk1"],
        #   },
        #   "layerwise": true
        # }
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not lmcache_adapter:
            return _create_error_response(
                {
                    "error": "LMCache adapter unavailable",
                    "message": "LMCache adapter not configured.",
                },
                503,
            )

        if not slot_mapping:
            return _create_error_response(
                {
                    "error": "Missing parameters",
                    "message": "slot_mapping parameter is required",
                },
                400,
            )

        # Parse slot_mapping from mixed format string
        # (supports single numbers and ranges)
        slot_indices, error_info = parse_mixed_slot_mapping(slot_mapping)
        if error_info:
            return _create_error_response(error_info, 400)

        # slot_indices is guaranteed to be non-None when error_info is None
        assert slot_indices is not None

        # Validate slot indices are within valid range
        if lmcache_adapter.kv_caches:
            # Get the first kv_tensor to check dimensions
            first_kv_tensor = next(iter(lmcache_adapter.kv_caches.values()))
            # Calculate total slots: num_blocks * block_size
            # For different formats:
            # - MHA (5D): [2, num_blocks, block_size, num_heads, head_size]
            # - MLA (3D): [num_blocks, block_size, head_size]
            # - 4D: [num_blocks, block_size, num_heads, head_size]
            ndim = first_kv_tensor.ndim
            if ndim == 5:
                # MHA: [2, num_blocks, block_size, num_heads, head_size]
                total_slots = first_kv_tensor.shape[1] * first_kv_tensor.shape[2]
            elif ndim == 3:
                # MLA: [num_blocks, block_size, head_size]
                total_slots = first_kv_tensor.shape[0] * first_kv_tensor.shape[1]
            elif ndim == 4:
                # 4D: [num_blocks, block_size, num_heads, head_size]
                total_slots = first_kv_tensor.shape[0] * first_kv_tensor.shape[1]
            else:
                # Fallback
                reshaped = first_kv_tensor.view(-1, *first_kv_tensor.shape[2:])
                total_slots = reshaped.shape[0]

            # Check each slot index
            invalid_indices = []
            for slot_idx in slot_indices:
                if slot_idx < 0 or slot_idx >= total_slots:
                    invalid_indices.append(slot_idx)

            if invalid_indices:
                return _create_error_response(
                    {
                        "error": "Invalid slot indices",
                        "message": (
                            "Slot indices out of bounds: %s. Valid range: 0 to %d"
                        )
                        % (invalid_indices, total_slots - 1),
                    },
                    400,
                )
        else:
            return _create_error_response(
                {
                    "error": "kv_caches not available",
                    "message": "kv_caches is empty or not initialized",
                },
                404,
            )

        # Validate chunk_size if provided
        if chunk_size is not None and chunk_size <= 0:
            return _create_error_response(
                {
                    "error": "Invalid chunk_size",
                    "message": "chunk_size must be a positive integer",
                },
                400,
            )

        # Get checksums from the adapter asynchronously to not block the loop
        loop = asyncio.get_running_loop()
        checksums_result = await loop.run_in_executor(
            None,  # Uses default ThreadPoolExecutor
            lambda: compute_kvcache_checksums(
                lmcache_adapter, slot_indices, chunk_size, layerwise
            ),
        )

        if checksums_result is None:
            return _create_error_response(
                {
                    "error": "Failed to compute checksums",
                    "message": "kv_caches not available or empty",
                },
                500,
            )

        # Compute slot mapping ranges using compress_slot_mapping
        slot_mapping_ranges = compress_slot_mapping(slot_indices)

        response_data: dict[str, Any] = {
            "status": "success",
            "slot_mapping_ranges": slot_mapping_ranges,
        }

        # Include chunk checksums
        response_data["chunk_size"] = checksums_result.get("chunk_size")
        response_data["num_chunks"] = checksums_result.get("num_chunks")
        response_data["chunk_checksums"] = checksums_result.get("chunk_checksums")
        response_data["layerwise"] = layerwise

        return PlainTextResponse(
            content=json.dumps(response_data, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Failed to compute kvcache checksums: %s", str(e))
        return _create_error_response(
            {"error": "Failed to compute checksums", "message": str(e)},
            500,
        )


@router.post("/cache/kvcache/record_slot")
async def kvcache_record_slot(
    request: Request,
    enabled: Optional[str] = None,
):
    """Enable or disable KVCache Check slot_mapping logging.

    This endpoint controls whether the KVCache Check logs (slot_mapping info)
    are printed when store/retrieve operations are performed.

    Args:
        request (Request): The FastAPI request object containing application state.
        enabled (Optional[str], optional): "true" to enable logging, "false" to
            disable. Defaults to None.

    Returns:
        PlainTextResponse: A JSON response containing the current logging status.

    Example:
        ```bash
        # Enable logging
        curl -X POST "http://localhost:8000/cache/kvcache/record_slot?enabled=true"

        # Disable logging
        curl -X POST "http://localhost:8000/cache/kvcache/record_slot?enabled=false"

        # Check current status
        curl -X POST "http://localhost:8000/cache/kvcache/record_slot"
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not lmcache_adapter:
            return _create_error_response(
                {
                    "error": "LMCache adapter unavailable",
                    "message": "LMCache adapter not configured.",
                },
                503,
            )

        # Get current status from lmcache_engine
        lmcache_engine = lmcache_adapter.lmcache_engine
        current_status = getattr(lmcache_engine, "kvcache_check_log_enabled", False)

        # Update status if enabled parameter is provided
        if enabled is not None:
            enabled_lower = enabled.lower()
            if enabled_lower == "true":
                lmcache_engine.kvcache_check_log_enabled = True
                current_status = True
                logger.info("KVCache Check logging enabled")
            elif enabled_lower == "false":
                lmcache_engine.kvcache_check_log_enabled = False
                current_status = False
                logger.info("KVCache Check logging disabled")
            else:
                return _create_error_response(
                    {
                        "error": "Invalid parameter",
                        "message": "enabled must be 'true' or 'false'",
                    },
                    400,
                )

        response_data = {
            "status": "success",
            "kvcache_check_log_enabled": current_status,
        }

        return PlainTextResponse(
            content=json.dumps(response_data, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Failed to set kvcache record slot status: %s", str(e))
        return _create_error_response(
            {"error": "Failed to set record slot status", "message": str(e)},
            500,
        )


@router.get("/cache/kvcache/info")
async def kvcache_info(request: Request):
    """Get information about the current kvcaches.

    Returns information about the kvcaches structure including layer names,
    shapes, and device information.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response containing kvcache information.
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not lmcache_adapter:
            return _create_error_response(
                {
                    "error": "LMCache adapter unavailable",
                    "message": "LMCache adapter not configured.",
                },
                503,
            )

        kv_caches = getattr(lmcache_adapter, "kvcaches", None)
        if not kv_caches:
            return _create_error_response(
                {
                    "error": "kv_caches not available",
                    "message": "kv_caches is empty or not initialized",
                },
                404,
            )

        layers_info: dict = {}
        for layer_name, kv_tensor in kv_caches.items():
            layers_info[layer_name] = {
                "shape": list(kv_tensor.shape),
                "dtype": str(kv_tensor.dtype),
                "device": str(kv_tensor.device),
            }

        info = {
            "status": "success",
            "num_layers": len(kv_caches),
            "layers": layers_info,
        }

        return PlainTextResponse(
            content=json.dumps(info, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Failed to get kvcache info: %s", str(e))
        return _create_error_response(
            {"error": "Failed to get kvcache info", "message": str(e)},
            500,
        )
