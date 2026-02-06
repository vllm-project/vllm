# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading

from typing import TYPE_CHECKING, Dict, Optional, List, Callable

import ray
from ray.util.placement_group import PlacementGroup

from vllm import envs
from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType

from vllm.v1.engine.utils import EngineZmqAddresses, CoreEngineActorManager
from vllm.v1.executor import Executor
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.logger import init_logger

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

