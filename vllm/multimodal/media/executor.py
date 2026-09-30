# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
from concurrent.futures import ThreadPoolExecutor

from vllm import envs

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)
