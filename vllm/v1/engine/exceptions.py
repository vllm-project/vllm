# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM V1 引擎异常定义模块。

本模块定义了 vLLM V1 引擎使用的异常类。
"""


class EngineGenerateError(Exception):
    """当 AsyncLLM.generate() 失败时抛出。可恢复。

    此异常表示生成过程中出现了错误，但该错误可能是暂时的，
    客户端可以选择重试。
    """

    pass


class EngineDeadError(Exception):
    """当 EngineCore 死亡时抛出。不可恢复。

    此异常表示引擎核心进程已经死亡，无法继续处理请求。
    客户端应该停止发送新请求并进行清理。
    """

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # 在与 LLMEngine 一起使用时，通过抑制不相关的 ZMQError 使堆栈跟踪更清晰
        self.__suppress_context__ = suppress_context
