# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""请求队列模块。

本模块实现了请求队列的抽象和具体实现，负责：
- 定义请求队列的抽象接口
- 提供 FCFS（先来先服务）队列实现
- 提供优先级队列实现

主要类：
- SchedulingPolicy: 调度策略枚举（FCFS、PRIORITY）
- RequestQueue: 请求队列抽象基类
- FCFSRequestQueue: 先来先服务队列
- PriorityRequestQueue: 优先级队列

主要函数：
- create_request_queue: 根据策略创建请求队列
"""

import heapq
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """调度策略枚举。

    定义两种调度策略：
    - FCFS: 先来先服务（First-Come-First-Served）
    - PRIORITY: 优先级调度（priority 值小的先处理）
    """

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """请求队列抽象基类。

    定义了请求队列的标准接口，支持以下操作：
    - add_request: 根据策略添加请求
    - pop_request: 根据策略弹出请求
    - peek_request: 查看队首请求但不移除
    - prepend_request: 在队首添加请求
    - remove_request: 移除指定请求
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """根据策略将请求添加到队列。

        Args:
            request: 要添加的请求
        """
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """根据策略从队列弹出一个请求。

        Returns:
            弹出的请求
        """
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """查看队首请求但不移除。

        Returns:
            队首请求
        """
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """在队首添加请求。

        Args:
            request: 要添加的请求
        """
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """将另一个队列的所有请求添加到队首。

        Args:
            requests: 要添加的请求队列
        """
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """从队列移除指定请求。

        Args:
            request: 要移除的请求
        """
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列移除多个指定请求。

        Args:
            requests: 要移除的请求列表
        """
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """检查队列是否有请求。

        Returns:
            如果队列非空则返回 True
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """获取队列中的请求数量。

        Returns:
            请求数量
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """根据策略遍历队列。

        Yields:
            队列中的请求
        """
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """先来先服务队列实现。

    使用 deque 实现，支持双端队列操作。
    新请求添加到队尾，从队头弹出请求。
    """

    def add_request(self, request: Request) -> None:
        """根据 FCFS 策略将请求添加到队列（队尾）。

        Args:
            request: 要添加的请求
        """
        self.append(request)

    def pop_request(self) -> Request:
        """根据 FCFS 策略从队列弹出请求（队头）。

        Returns:
            弹出的请求
        """
        return self.popleft()

    def peek_request(self) -> Request:
        """查看队首请求但不移除。

        Returns:
            队首请求

        Raises:
            IndexError: 如果队列为空
        """
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """在队首添加请求。

        Args:
            request: 要添加的请求
        """
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将另一个队列的所有请求添加到队首。

        注意：请求以与 `requests` 队列中出现的相反顺序添加。

        Args:
            requests: 要添加的请求队列
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """从队列移除指定请求。

        Args:
            request: 要移除的请求
        """
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列移除多个指定请求。

        对于单个请求使用 fast path（原地修改），
        对于多个请求使用列表推导（创建新列表）。

        Args:
            requests: 要移除的请求列表
        """
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque 不支持原地过滤，所以需要 clear 和 extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """检查队列是否有请求。

        Returns:
            如果队列非空则返回 True
        """
        return len(self) > 0

    def __len__(self) -> int:
        """获取队列中的请求数量。

        Returns:
            请求数量
        """
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """根据 FCFS 策略遍历队列。

        Yields:
            队列中的请求
        """
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """优先级队列实现。

    使用堆（heap）实现，尊重 Request 类中定义的排序：
    - priority 值较小的请求先处理
    - 如果 priority 相同，则 arrival_time 较早的请求先处理
    """

    def __init__(self) -> None:
        """初始化优先级队列。"""
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """根据优先级策略将请求添加到队列。

        Args:
            request: 要添加的请求
        """
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """根据优先级策略从队列弹出请求。

        Returns:
            弹出的请求

        Raises:
            IndexError: 如果队列为空
        """
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """查看队首请求但不移除。

        Returns:
            队首请求

        Raises:
            IndexError: 如果队列为空
        """
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """在队首添加请求。

        注意：在优先级队列中，没有"队首"的概念。
        请求按照 (priority, arrival_time) 排序。

        Args:
            request: 要添加的请求
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """将另一个队列的所有请求添加到队列。

        注意：在优先级队列中，没有"队首"的概念。
        请求按照 (priority, arrival_time) 排序。

        Args:
            requests: 要添加的请求队列
        """
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """从队列移除指定请求。

        Args:
            request: 要移除的请求
        """
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """从队列移除多个指定请求。

        Args:
            requests: 要移除的请求列表
        """
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """检查队列是否有请求。

        Returns:
            如果队列非空则返回 True
        """
        return bool(self._heap)

    def __len__(self) -> int:
        """获取队列中的请求数量。

        Returns:
            请求数量
        """
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """根据优先级策略遍历队列。

        Yields:
            队列中的请求（按优先级排序）
        """
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """根据调度策略创建请求队列。

    Args:
        policy: 调度策略

    Returns:
        创建的请求队列

    Raises:
        ValueError: 如果策略未知
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
