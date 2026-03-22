# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus 多进程支持模块。

本模块提供 Prometheus 多进程模式的支持功能，负责：
- 设置多进程 Prometheus 目录
- 获取合适的 Prometheus 注册表
- 注销 vLLM 指标
- 关闭 Prometheus 服务

主要函数：
- setup_multiprocess_prometheus: 设置多进程 Prometheus
- get_prometheus_registry: 获取 Prometheus 注册表
- unregister_vllm_metrics: 注销 vLLM 指标
- shutdown_prometheus: 关闭 Prometheus
"""

import os
import tempfile

from prometheus_client import REGISTRY, CollectorRegistry, multiprocess

from vllm.logger import init_logger

logger = init_logger(__name__)

# Prometheus 多进程临时目录全局变量
_prometheus_multiproc_dir: tempfile.TemporaryDirectory | None = None


def setup_multiprocess_prometheus():
    """设置 Prometheus 多进程目录（如果尚未配置）。

    如果环境中没有设置 PROMETHEUS_MULTIPROC_DIR，
    则创建一个临时目录用于 Prometheus 多进程指标收集。

    注意：多进程模式下需要特别注意指标清理，
    否则可能导致指标不准确。
    """
    global _prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        # 为 Prometheus 创建临时目录
        # 注意：全局 TemporaryDirectory 会在退出时自动清理
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
        logger.debug(
            "Created PROMETHEUS_MULTIPROC_DIR at %s", _prometheus_multiproc_dir.name
        )
    else:
        logger.warning(
            "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
            "This directory must be wiped between vLLM runs or "
            "you will find inaccurate metrics. Unset the variable "
            "and vLLM will properly handle cleanup."
        )


def get_prometheus_registry() -> CollectorRegistry:
    """获取合适的 Prometheus 注册表。

    根据多进程配置返回相应的注册表。
    在多进程模式下，需要使用 MultiProcessCollector 来合并各进程的指标。

    Returns:
        CollectorRegistry: Prometheus 注册表
    """
    if os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None:
        logger.debug("Using multiprocess registry for prometheus metrics")
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry

    return REGISTRY


def unregister_vllm_metrics():
    """从 Prometheus 注册表注销所有现有的 vLLM 指标收集器。

    此函数在以下场景很有用：
    - 测试：避免多次测试运行之间指标重复注册
    - CI/CD：确保每次运行的指标独立
    - 多进程模式：需要从全局注册表注销指标

    通过遍历注册表中的所有收集器，
    移除名称包含 "vllm" 的收集器。
    """
    registry = REGISTRY
    # 注销现有的 vLLM 收集器
    for collector in list(registry._collector_to_names):
        if hasattr(collector, "_name") and "vllm" in collector._name:
            registry.unregister(collector)


def shutdown_prometheus():
    """关闭 Prometheus 指标服务。

    标记当前进程的指标为已失效，
    以便在多进程模式下正确清理。
    """

    path = _prometheus_multiproc_dir
    if path is None:
        return
    try:
        pid = os.getpid()
        multiprocess.mark_process_dead(pid, path)
        logger.debug("Marked Prometheus metrics for process %d as dead", pid)
    except Exception as e:
        logger.error("Error during metrics cleanup: %s", str(e))
