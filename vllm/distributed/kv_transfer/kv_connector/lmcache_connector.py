"""
    This file implements a simple torch distributed connector by 2 classes:
    - `TorchDistributedPipe`: a tensor transmission pipe between P/D instance,
      using `torch.distributed`
    - `TorchDistributedConnector`: a torch distributed connector between P/D 
      instance, implemented on top of `TorchDistributedPipe`
"""
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Union

import torch
from torch.distributed import Backend

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import lmcache
except ModuleNotFoundError as e:
    logger.error("LMcache not installed, please install LMCache.")
    raise e


class LMCacheConnector(KVConnectorBase):
    
    pass