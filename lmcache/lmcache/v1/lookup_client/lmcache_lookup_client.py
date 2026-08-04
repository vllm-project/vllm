# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import json
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.rpc.transport import (
        RpcClientTransport,
        RpcServerTransport,
    )

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """
    Lookup client that communicates with a lookup server
    via an injected RpcClientTransport.

    The client is decoupled from the underlying communication
    mechanism. The transport layer handles connection management,
    retries, and error recovery.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some
        workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup
          server on all workers
        - if lookup_server_worker_ids is [0], start lookup
          server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start
          lookup server on worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: "LMCacheEngineConfig",
        metadata: "LMCacheMetadata",
        transport: "RpcClientTransport",
    ):
        self.config = config
        self.transport = transport

        # NOTE: map from lookup_id (i.e., req_id) to
        # req's status.
        # int indicates number of hit tokens.
        # The assumption here is that once a request is
        # looked up, the following lookups of the same
        # request must have the same result.
        self.reqs_status: dict[str, int] = {}

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        "-1 means not found;
        None means ongoing; (not supported in sync client)
        int >= 0 means number of hit tokens
        """
        return self.reqs_status.get(lookup_id, -1)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = json.dumps(request_configs)

        # NOTE(Jiayi): We cannot only send hashes when
        # blending enabled because the blender need the
        # input embedding.
        if not self.enable_blending:
            hashes = []
            offsets = []

            for (
                start,
                end,
                key,
            ) in self.token_database.process_tokens(token_ids, make_key=False):
                hashes.append(key)
                offsets.append(end - start)

            # if the token database returns no hashes,
            # return 0
            if not hashes:
                return 0

            msg_buf = [
                hashes,
                offsets,
                lookup_id,
                request_configs_str,
            ]
        else:
            # Convert token_ids to a plain list for msgpack serialization
            # (vLLM 0.18+ may pass ConstantList which msgspec can't encode)
            if isinstance(token_ids, torch.Tensor):
                serializable_ids = token_ids.tolist()
            elif not isinstance(token_ids, list):
                serializable_ids = list(token_ids)
            else:
                serializable_ids = token_ids
            msg_buf = [
                serializable_ids,
                lookup_id,
                request_configs_str,
            ]

        responses = self.transport.send_and_recv_all(msg_buf)

        # Transport returns empty list on failure
        if not responses:
            return 0

        results = [int.from_bytes(resp, "big") for resp in responses]

        assert len(results) == self.transport.world_size
        if len(set(results)) > 1:
            logger.warning(
                "Lookup results (number of hit tokens) "
                "differ across (TP and PP) ranks: %s.",
                results,
            )
        # NOTE: it is possible that the number of hit
        # tokens is different across (TP and PP) ranks,
        # so we can use the minimum value.
        num_hit_toks = min(results)
        self.reqs_status[lookup_id] = num_hit_toks

        return num_hit_toks

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports
        producer kvcache reuse"""
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        self.transport.close()


class LMCacheLookupServer:
    """Lookup server that handles lookup requests using
    LMCacheEngine, with an injected RpcServerTransport."""

    def __init__(
        self,
        lmcache_engine: "LMCacheEngine",
        metadata: "LMCacheMetadata",
        transport: "RpcServerTransport",
    ):
        self.transport = transport
        self.lmcache_engine = lmcache_engine
        self.running = True
        self.enable_blending = lmcache_engine.config.enable_blending

        def process_request():
            while self.running:
                try:
                    result = self.transport.recv_request()
                    if result is None:
                        continue

                    identity, data_frames = result

                    # Validate frame structure
                    if len(data_frames) < 3:
                        logger.warning("Malformed request received: not enough frames.")
                        continue

                    # Validate and decode lookup_id
                    lookup_id_bytes = data_frames[-2]
                    request_configs_bytes = data_frames[-1]

                    if not isinstance(lookup_id_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: lookup_id is not bytes or str."
                        )
                        continue

                    if not isinstance(request_configs_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: "
                            "request_configs is not bytes or str."
                        )
                        continue

                    # Decode to strings
                    if isinstance(lookup_id_bytes, bytes):
                        lookup_id = lookup_id_bytes.decode("utf-8")
                    else:
                        lookup_id = lookup_id_bytes

                    if isinstance(request_configs_bytes, bytes):
                        request_configs_str = request_configs_bytes.decode("utf-8")
                    else:
                        request_configs_str = request_configs_bytes

                    request_configs = (
                        json.loads(request_configs_str) if request_configs_str else None
                    )

                    if not self.enable_blending:
                        hashes = data_frames[0]
                        offsets = data_frames[1]
                        lookup_result = self.lmcache_engine.lookup(
                            hashes=hashes,
                            offsets=offsets,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                        )
                    else:
                        tokens = data_frames[0]
                        lookup_result = self.lmcache_engine.lookup(
                            tokens=tokens,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                        )
                    response = lookup_result.to_bytes(4, "big")
                    self.transport.send_response(identity, response)
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON in lookup request: %s", e)
                except UnicodeDecodeError as e:
                    logger.error("Error decoding UTF-8 in lookup request: %s", e)
                except Exception as e:
                    logger.error("Error processing lookup request: %s", e)

        logger.info("lmcache lookup server started")
        self.thread = threading.Thread(
            target=process_request,
            daemon=True,
            name="lookup-server-thread",
        )
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        # Stop the processing thread first
        self.running = False

        # Wait for thread to finish with timeout
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Lookup server thread did not terminate gracefully")

        # Close transport after thread is stopped
        self.transport.close()
