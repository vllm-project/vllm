# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 卸载规范工厂模块。

本模块提供卸载规范的注册和创建功能，负责：
- 注册各种卸载规范类型
- 支持懒加载模块和类
- 根据配置创建相应的卸载规范实例

主要类：
- OffloadingSpecFactory: 卸载规范工厂类
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.spec import OffloadingSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class OffloadingSpecFactory:
    """卸载规范工厂类。

    提供规范的注册和创建功能，支持懒加载机制。
    通过注册表管理各种卸载规范类型，允许用户通过配置
    指定使用哪种规范实现。

    Attributes:
        _registry: 规范名称到加载器的映射字典
    """

    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(cls, name: str, module_path: str, class_name: str) -> None:
        """注册一个规范，使用懒加载的模块和类名。

        Args:
            name: 规范名称
            module_path: 模块路径
            class_name: 类名

        Raises:
            ValueError: 如果该名称的连接器已注册
        """
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[OffloadingSpec]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_spec(
        cls,
        config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ) -> OffloadingSpec:
        """根据配置创建卸载规范实例。

        从 kv_transfer_config 中读取规范名称，从注册表中
        获取对应的规范类并实例化。如果注册表中没有找到，
        则尝试从指定的模块路径动态加载。

        Args:
            config: vLLM 配置
            kv_cache_config: KV 缓存配置

        Returns:
            创建的卸载规范实例

        Raises:
            ValueError: 如果指定的规范类型不受支持
        """
        kv_transfer_config = config.kv_transfer_config
        assert kv_transfer_config is not None
        extra_config = kv_transfer_config.kv_connector_extra_config
        spec_name = extra_config.get("spec_name", "CPUOffloadingSpec")
        if spec_name in cls._registry:
            spec_cls = cls._registry[spec_name]()
        else:
            spec_module_path = extra_config.get("spec_module_path")
            if spec_module_path is None:
                raise ValueError(f"Unsupported spec type: {spec_name}")
            spec_module = importlib.import_module(spec_module_path)
            spec_cls = getattr(spec_module, spec_name)
        assert issubclass(spec_cls, OffloadingSpec)
        logger.info("Creating offloading spec with name: %s", spec_name)
        return spec_cls(config, kv_cache_config)


# 注册各种规范
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec", "vllm.v1.kv_offload.cpu", "CPUOffloadingSpec"
)
