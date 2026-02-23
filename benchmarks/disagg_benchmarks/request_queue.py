# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import deque


class RequestQueue:
    """Request queue manager with concurrency control"""

    def __init__(self, max_concurrent, max_queue_size):
        # Maximum concurrent requests
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size  # Maximum queue size
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = deque()  # Request queue
        self.queue_size = 0  # Current queue size
        self.lock = asyncio.Lock()  # Sync queue Lock

    async def enqueue(self, task):
        """Add a request task to the queue"""
        async with self.lock:
            if self.queue_size >= self.max_queue_size:
                return False

            self.queue.append(task)
            self.queue_size += 1
            return True

    async def process(self):
        """Process queued requests using semaphore for concurrency control"""
        while True:
            if self.queue:
                async with self.semaphore, self.lock:
                    task = self.queue.popleft()
                    self.queue_size -= 1
                    await task
            await asyncio.sleep(0.01)  # Yield control to event loop
