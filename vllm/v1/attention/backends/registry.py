# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

# [CN] 文件总览：注意力后端的枚举清单与注册表
# [CN] 职责：集中维护「后端名字 <-> 实现类路径」的映射，并允许在运行时被覆盖。
# [CN] 链路：selector.py -> AttentionBackendEnum.<NAME>.get_class() -> 具体 Backend 类
# [CN]       反向也有：外部插件 register_backend(...) 把实现类挂回枚举成员上。
# [CN] 设计要点：
# [CN]   1. 枚举的 value 只是**默认**类路径，真正的解析走 _ATTN_OVERRIDES 覆盖表 ——
# [CN]      第三方不必 fork vLLM 就能替换内置后端。
# [CN]   2. 这里登记了约 30 个后端（含 ROCm / XPU / CPU / TPU），与实际可用性无关：
# [CN]      能否用取决于当前 platform 的 get_attn_backend_cls() 是否返回它。
# [CN]   3. TORCH_SDPA 的 value 是空串，注释指明只用于 ViT —— 它是一个路由名而非通用后端。
# [CN]   4. CUSTOM 的 value 是 None：占位符，必须先 register_backend 才能取到路径，
# [CN]      否则报「must be registered before use」。用 None 而非空串是为了不和别的成员别名冲突。

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)


# [CN] 这个元类只做一件小事：把 KeyError 换成带「合法取值清单」的 ValueError。
# [CN] 用户配置里写错 backend 名时，错误信息直接告诉他有哪些可选项，省一次查代码。

class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# [CN] 全部 attention 后端。取类请走 .get_class()（会查覆盖表），不要直接读 .value
# [CN] —— 直接读 value 会绕过运行时注册，拿到的是默认实现而不是用户替掉的那个。

class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    # [CN] 以下每个成员的 value 是「默认实现类路径」，可被 register_backend 覆盖。

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    TRITON_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.triton_attn_diffkv.TritonAttentionDiffKVBackend"
    )
    # [CN] 以下一组是 AMD ROCm 平台专用（含 AITER 融合算子与 MLA 变体）。

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
    # [CN] NVIDIA 上的 FlashInfer 系列（含普通版与各类 MLA / sparse 变体）。

    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.tokenspeed_mla.TokenspeedMLABackend"
    )
    FLASHINFER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseTRTLLMBackend"
    )
    FLASHINFER_MLA_SPARSE_SM120 = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseSM120Backend"
    )
    FLASHINFER_MLA_SPARSE_SM90 = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90."
        "FlashInferMLASparseSM90Backend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    # [CN] MLA（Multi-head Latent Attention，DeepSeek 系）专用后端的几个实现版本。

    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    # DeepSeek V4 sparse MLA backends (model-driven; selected via the V4 layer).
    FLASHMLA_SPARSE_DSV4 = (
        "vllm.models.deepseek_v4.sparse_mla.DeepseekV4FlashMLABackend"
    )
    FLASHINFER_MLA_SPARSE_DSV4 = (
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."
        "DeepseekV4FlashInferMLASparseBackend"
    )
    ROCM_FLASHMLA_SPARSE_DSV4 = (
        "vllm.models.deepseek_v4.amd.rocm.DeepseekV4ROCMAiterMLASparseBackend"
    )
    B12X = "vllm.v1.attention.backends.b12x.B12xPagedAttentionBackend"
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    FLASH_ATTN_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseBackend"
    )
    MINIMAX_M3_SPARSE = (
        "vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend"
    )
    CUTLASS_MSA = (
        "vllm.models.minimax_m3.nvidia.sparse_attention_msa."
        "MiniMaxM3SparseCutlassBackend"
    )
    TRITON_MSA = (
        "vllm.models.minimax_m3.nvidia.sparse_attention_msa."
        "MiniMaxM3SparseTritonBackend"
    )
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    # HPC Attention Backend:
    # powered by operators from https://github.com/Tencent/hpc-ops.
    # Only supported on NVIDIA Hopper GPUs (e.g. H20, H200),
    # currently limited to the Hy3 model,
    # and requires a block size of 64.
    HPC_ATTN = "vllm.v1.attention.backends.hpc_attn.HpcAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = (
        "vllm.v1.attention.backends.rocm_aiter_unified_attn."
        "RocmAiterUnifiedAttentionBackend"
    )
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    # [CN] CPU 侧后端，主要用于本地调试与小模型 CPU 推理。

    CPU_MLA = "vllm.v1.attention.backends.mla.cpu_mla.CPUMLABackend"
    AMX_MLA = "vllm.v1.attention.backends.mla.amx_mla.AMXMLABackend"
    # [CN] 量化 KV 与其他平台专用后端（B12X / HPC / FLEX / CPU / AMX 等）。

    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    # [CN] CUSTOM 是第三方占位：value 为 None，未注册就取路径会报错，
    # [CN] 这样「还没注册」和「注册成空串」不会混为一谈。

    CUSTOM = None

    # [CN] 解析真实类路径：先查覆盖表，未被覆盖才回落到枚举自带的默认 value。
    # [CN] include_classname=False 时剥掉最后一段，返回模块路径（某些 lazy import 场景需要）。

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
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

    # [CN] 通过 resolve_obj_by_qualname 完成字符串 -> 类的懒加载：
    # [CN] 不在 import 本模块时就把所有后端（及其 CUDA 扩展）拉进来。

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


# [CN] Mamba / 短卷积 / 线性注意力这类「无 KV cache」后端的独立枚举。
# [CN] 结构与上面完全镜像：同样的元类、同样的覆盖表机制，只是选择时不传 AttentionSelectorConfig。

class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported mamba attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    # [CN] mamba / 线性注意力后端清单：这类层没有 KV cache，只有循环 state 或卷积状态。

    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    # [CN] mamba 侧的第三方占位，语义与 AttentionBackendEnum.CUSTOM 相同。

    CUSTOM = None

    # [CN] 与 AttentionBackendEnum.get_path 完全等价，只是作用于 mamba 侧的 _MAMBA_ATTN_OVERRIDES。

    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
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

    # [CN] mamba 侧的懒加载，查的是同样的 _MAMBA 覆盖表。

    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden.

        Returns:
            True if the backend has a registered override
        """
        return self in _MAMBA_ATTN_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)


# [CN] 两张覆盖表：key 是枚举成员，value 是「被替换成的类路径」。属于全局可变状态，
# [CN] 统一由 register_backend / clear_override 维护；测试里改完务必还原，否则会污染同进程的其他用例。

_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}


# [CN] 覆盖/注册入口，支持两种用法：
# [CN]   装饰器形式 register_backend(Enum.X)  —— 从 cls 自动推导 __module__.__qualname__
# [CN]   直接调用 register_backend(Enum.X, 'path.Cls') —— 不依赖装饰语法，可在任意时机注册
# [CN] 前者返回 decorator，后者返回恒等函数 lambda x: x —— 这样两种方式的返回值
# [CN] 都能安全地贴在类定义上，调用方不用区分。

def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

    Examples:
        # Override an existing attention backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Override an existing mamba attention backend
        @register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # Register a custom third-party attention backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    # [CN] 装饰器形式：把装饰时的 cls 位置写入覆盖表，然后原样返回 —— 不包裹、不改变类行为。

    def decorator(cls: type) -> type:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    # [CN] 显式路径形式：登记字符串即可，典型用途是插件在自己的 setup 阶段注册实现类。

    if class_path is not None:
        if is_mamba:
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
