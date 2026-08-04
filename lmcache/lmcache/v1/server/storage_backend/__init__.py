# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.server.storage_backend.abstract_backend import LMSBackendInterface
from lmcache.v1.server.storage_backend.local_backend import LMSLocalBackend

logger = init_logger(__name__)


def CreateStorageBackend(device: str) -> LMSBackendInterface:
    match device:
        case "cpu":
            # cpu only
            logger.info("Initializing cpu-only cache server")
            return LMSLocalBackend()
        case _:
            raise ValueError(f"Unsupported device: {device}")
        # TODO(Jiayi): please implement hierarchical remote storage
        # case _:
        #    logger.info("Initializing disk-only cache server")
        #    return LMSLocalDiskBackend(path=device)
