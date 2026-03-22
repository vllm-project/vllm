# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executor 模块入口。

本模块是 vLLM V1 执行器的入口文件，提供模型执行相关的
抽象类和具体实现。

主要类：
- Executor: 执行器抽象基类，定义执行器接口
- UniProcExecutor: 单进程执行器，用于单卡或调试场景

执行器负责在一个或多个设备上执行模型，支持分布式执行。
"""

from .abstract import Executor
from .uniproc_executor import UniProcExecutor

__all__ = ["Executor", "UniProcExecutor"]
