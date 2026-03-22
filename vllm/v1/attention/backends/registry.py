# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention 后端注册表模块。

本模块实现了注意力后端的枚举和注册机制，负责：
- 定义所有支持的注意力后端枚举
- 提供后端类的动态解析和注册功能
- 支持 Mamba 后端注册
- 提供友好的错误提示

主要类：
- _AttentionBackendEnumMeta: 注意力后端枚举元类
- AttentionBackendEnum: 注意力后端枚举
- MambaAttentionBackendEnum: Mamba 注意力后端枚举

主要函数：
- register_backend: 注册自定义注意力后端
- register_mamba_backend: 注册自定义 Mamba 后端
"""

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


class _AttentionBackendEnumMeta(EnumMeta):
    """注意力后端枚举元类。

    为 AttentionBackendEnum 提供更好的错误提示。
    """

    def __getitem__(cls, name: str):
        """通过名称获取后端，提供友好的错误提示。

        Args:
            name: 后端名称

        Returns:
            对应的枚举值

        Raises:
            ValueError: 如果名称无效，会列出所有有效选项
        """
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """注意力后端枚举类。

    枚举所有支持的注意力后端。枚举值是默认的类路径，
    可以通过 register_backend() 在运行时覆盖。

    要获取实际的后端类（尊重覆盖），使用：backend.get_class()

    支持的后端包括：
    - FLASH_ATTN: Flash Attention
    - FLASH_ATTN_DIFFKV: Flash Attention DiffKV
    - TRITON_ATTN: Triton Attention
    - ROCM_ATTN: ROCm Attention
    - ROCM_AITER_MLA: ROCm Aiter MLA
    - FLASHINFER: FlashInfer
    - FLASHMLA: Flash MLA
    - CPU_ATTN: CPU Attention
    - 等等...
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_TRITON_MLA = (
        "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    )
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    )
    ROCM_AITER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    )
    XPU_MLA_SPARSE = "vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend"
    TORCH_SDPA = ""  # this tag is only used for ViT
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    FLASHINFER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseBackend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    TREE_ATTN = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = (
        "vllm.v1.attention.backends.rocm_aiter_unified_attn."
        "RocmAiterUnifiedAttentionBackend"
    )
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """获取后端的类路径（尊重覆盖）。

        Args:
            include_classname: 是否包含类名（False 则只返回模块路径）

        Returns:
            完整的类路径字符串

        Raises:
            ValueError: 如果使用了未注册的 CUSTOM 后端
        """
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """获取后端类（尊重覆盖）。

        Returns:
            后端类

        Raises:
            ImportError: 如果无法导入后端类
            ValueError: 如果使用了未注册的 CUSTOM 后端
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """检查此外端是否有覆盖。

        Returns:
            如果后端有注册的覆盖则返回 True
        """
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:
        """清除此外端的任何覆盖，恢复为默认值。"""
        _ATTN_OVERRIDES.pop(self, None)


class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Mamba 注意力后端枚举类。

    枚举所有支持的 Mamba 注意力后端。枚举值是默认的类路径，
    可以通过 register_backend() 在运行时覆盖。

    支持的后端包括：
    - MAMBA1: Mamba1
    - MAMBA2: Mamba2
    - SHORT_CONV: Short Conv
    - LINEAR: Linear Attention
    - GDN_ATTN: GDN Attention
    """

    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:
        """获取后端的类路径（尊重覆盖）。

        Args:
            include_classname: 是否包含类名（False 则只返回模块路径）

        Returns:
            完整的类路径字符串

        Raises:
            ValueError: 如果使用了未注册的 CUSTOM 后端
        """
        path = _MAMBA_ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":
        """获取后端类（尊重覆盖）。

        Returns:
            后端类

        Raises:
            ImportError: 如果无法导入后端类
            ValueError: 如果使用了未注册的 CUSTOM 后端
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """检查此 backend 是否有覆盖。

        Returns:
            如果 backend 有注册的覆盖则返回 True
        """
        return self in _MAMBA_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """清除此后端的任何覆盖，恢复为默认值。"""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)


MAMBA_TYPE_TO_BACKEND_MAP = {
    "mamba1": MambaAttentionBackendEnum.MAMBA1.name,
    "mamba2": MambaAttentionBackendEnum.MAMBA2.name,
    "short_conv": MambaAttentionBackendEnum.SHORT_CONV.name,
    "linear_attention": MambaAttentionBackendEnum.LINEAR.name,
    "gdn_attention": MambaAttentionBackendEnum.GDN_ATTN.name,
    "custom": MambaAttentionBackendEnum.CUSTOM.name,
}


_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}


def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """注册或覆盖后端实现。

    可以作为装饰器使用，也可以直接调用。

    Args:
        backend: 要注册的 AttentionBackendEnum 成员
        class_path: 可选的类路径。如果未提供且作为装饰器使用，
                    将自动生成自类。
        is_mamba: 是否为 Mamba 后端

    Returns:
        如果 class_path 为 None 则返回装饰器函数，否则无操作

    Examples:
        # 覆盖现有的注意力后端
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            pass

        # 注册自定义后端
        register_backend(AttentionBackendEnum.CUSTOM,
            ...

        # 覆盖现有的 Mamba 注意力后端
        @register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # 注册自定义第三方后端
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # 直接注册
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    if class_path is not None:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
