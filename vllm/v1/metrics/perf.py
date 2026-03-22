# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""性能分析模块。

本模块实现了对 transformer 组件的分析 FLOPs/内存估算，
用于推导运行模型的 MFU（Model Flops Utilization）指标。

主要功能：
- 分析模型各组件（Attention、FFN、Unembed）的 FLOPs 和内存访问
- 支持量化配置的有效权重字节大小计算
- 提供每 GPU 的 FLOPs 和内存带宽估算
- Prometheus 指标集成

主要类：
- ExecutionContext: 执行上下文，汇总批次中请求的统计信息
- ComponentMetrics: 组件指标抽象基类
- AttentionMetrics: 注意力层指标
- FfnMetrics: 前馈网络层指标
- UnembedMetrics: 输出层指标
- ModelMetrics: 模型整体指标
- PerfStats: 性能统计数据
- PerfMetricsLogging: 性能指标日志记录
- PerfMetricsProm: Prometheus 性能指标

主要解析器：
- BaseConfigParser: 基础配置解析
- BaseAttentionConfigParser: 注意力配置解析
- AttentionQuantizationConfigParser: 注意力量化配置解析
- BaseFfnConfigParser: FFN 配置解析
- FfnParallelParser: FFN 并行配置解析
- FfnQuantizationConfigParser: FFN 量化配置解析
"""

import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import prometheus_client
import torch
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_dtype_size,
    get_kv_cache_torch_dtype,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


class InvalidComponent(Exception):
    """自定义异常，表示某个 ComponentMetric 不适用于给定的 VllmConfig。"""

    pass


# 量化方法名称到有效权重字节大小的映射
# 被 AttentionQuantizationConfigParser 和 FfnQuantizationConfigParser 使用
# 用于确定 FLOPs/内存估算的 weight_byte_size
#
# 注意：GPTQ 和 BitsAndBytes 等方法支持可变位宽（如 4 位和 8 位）。
# 默认使用 4 位（0.5 字节），因为这是最常见的配置。
_QUANT_WEIGHT_BYTE_SIZE: dict[str, float] = {
    # FP8 方法（每个权重 1 字节）
    "fp8": 1,
    "fbgemm_fp8": 1,
    "ptpc_fp8": 1,
    "fp_quant": 1,
    "modelopt": 1,
    "modelopt_mxfp8": 1,
    # FP4 / INT4 方法（每个权重 0.5 字节）
    "mxfp4": 0.5,
    "awq": 0.5,
    "awq_marlin": 0.5,
    "gptq": 0.5,
    "gptq_marlin": 0.5,
    "bitsandbytes": 0.5,
    "modelopt_fp4": 0.5,
    "petit_nvfp4": 0.5,
    "gguf": 0.5,
    "compressed-tensors": 0.5,
    "torchao": 0.5,
    "quark": 0.5,
    "moe_wna16": 0.5,
    "inc": 0.5,
    "cpu_awq": 0.5,
    "experts_int8": 1,
}


#### 基本数据类型 ####


@dataclass
class DebugPerfStats:
    """性能统计调试信息。

    Attributes:
        calc_duration: 计算这些统计信息花费的时间
        num_prefill_requests: 预填充请求数量
        num_decode_requests: 解码请求数量
        context_breakdown: 上下文细分统计
        num_flops_per_gpu_breakdown: 每 GPU FLOPs 细分
        num_read_bytes_per_gpu_breakdown: 每 GPU 读取字节数细分
        num_write_bytes_per_gpu_breakdown: 每 GPU 写入字节数细分
    """
    ## 调试指标计算的统计
    calc_duration: float = 0.0  # 计算这些统计信息花费的时间
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
    num_flops_per_gpu_breakdown: dict[str, int] | None = None
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None


@dataclass
class PerfStats:
    """性能统计数据。

    Attributes:
        num_flops_per_gpu: 每 GPU FLOPs
        num_read_bytes_per_gpu: 每 GPU 读取字节数
        num_write_bytes_per_gpu: 每 GPU 写入字节数
        debug_stats: 调试统计（可选）
    """
    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None


@dataclass
class ExecutionContext:
    """请求批次的执行上下文。

    此类汇总批次中多个请求的统计信息，
    分别跟踪预填充（prefill）和解码（decode）阶段。

    示例:
        - 批次包含一个完整预填充（2048 tokens）和一个解码（1 token, 8192 上下文）:
          ctx = ExecutionContext()
          ctx.add(2048, 2048, is_prefill=True)
          ctx.add(1, 8192, is_prefill=False)

    Attributes:
        num_prefill_requests: 预填充请求数量
        prefill_num_tokens: 预填充请求的 token 总数
        prefill_context_len: 预填充请求的上下文长度总和
        prefill_token_context_product: 预填充请求的 (num_tokens * context_len) 总和
        num_decode_requests: 解码请求数量
        decode_num_tokens: 解码请求的 token 总数
        decode_context_len: 解码请求的上下文长度总和
        decode_token_context_product: 解码请求的 (num_tokens * context_len) 总和
    """

    # 预填充阶段统计
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0  # 预填充请求的 num_tokens 总和
    prefill_context_len: int = 0  # 预填充请求的 context_len 总和
    prefill_token_context_product: int = 0  # 预填充请求的 (num_tokens * context_len) 总和

    # 解码阶段统计
    num_decode_requests: int = 0
    decode_num_tokens: int = 0  # 解码请求的 num_tokens 总和
    decode_context_len: int = 0  # 解码请求的 context_len 总和
    decode_token_context_product: int = 0  # 解码请求的 (num_tokens * context_len) 总和

    def add(self, num_tokens: int, context_len: int, is_prefill: bool) -> None:
        """将单个请求的统计信息添加到此批次上下文。

        Args:
            num_tokens: token 数量
            context_len: 上下文长度
            is_prefill: 是否为预填充阶段
        """
        if is_prefill:
            self.num_prefill_requests += 1
            self.prefill_num_tokens += num_tokens
            self.prefill_context_len += context_len
            self.prefill_token_context_product += num_tokens * context_len
        else:
            self.num_decode_requests += 1
            self.decode_num_tokens += num_tokens
            self.decode_context_len += context_len
            self.decode_token_context_product += num_tokens * context_len

    def total_num_tokens(self) -> int:
        """批次中所有请求的总 token 数。

        Returns:
            总 token 数
        """
        return self.prefill_num_tokens + self.decode_num_tokens

    def total_token_context_product(self) -> int:
        """所有请求的 (num_tokens * context_len) 总和。

        Returns:
            token-上下文乘积总和
        """
        return self.prefill_token_context_product + self.decode_token_context_product

    def num_logits_tokens(self) -> int:
        """需要 logits 计算（unembedding）的 token 数量。

        对于预填充，每个请求只需要最后一个 token 的 logits。
        对于解码，所有 token 都需要 logits。

        Returns:
            需要 logits 计算的 token 数量
        """
        return self.num_prefill_requests + self.decode_num_tokens

    @classmethod
    def from_single_request(
        cls, num_tokens: int, context_len: int, is_prefill: bool
    ) -> "ExecutionContext":
        """从单个请求创建 ExecutionContext。

        此便捷方法主要用于测试。

        Args:
            num_tokens: token 数量
            context_len: 上下文长度
            is_prefill: 是否为预填充阶段

        Returns:
            ExecutionContext 实例
        """
        ctx = cls()
        ctx.add(num_tokens, context_len, is_prefill)
        return ctx


class ParsedArgs:
    """解析参数的辅助类。

    使得解析器可以使用点符号访问/更新解析的参数。

    示例:
        args = ParsedArgs()
        args.x = 3
        args.y = args.x + 1
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def model_dump(self) -> dict[str, Any]:
        return vars(self).copy()


#### 抽象定义 ####


class Parser(Protocol):
    """解析器协议。

    解析器从 VllmConfig 中解析参数并传递给下一个解析器。
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析 vllm_config 并更新当前的 ParsedArgs 然后传递。

        如果解析器不适用于 vllm_config，则不执行任何操作。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        ...


class ParserChain:
    """解析器链。

    按顺序应用解析器链中的解析器。
    后面的解析器可能会覆盖前面解析器的结果，
    因此如果解析器不是互斥的，应该按适当的顺序链接。
    """

    def __init__(self, *parsers: Parser) -> None:
        self.parsers = list(parsers)

    def add_parser(self, parser: Parser) -> None:
        """添加解析器到链中。

        Args:
            parser: 要添加的解析器
        """
        self.parsers.append(parser)

    def parse(self, vllm_config: VllmConfig) -> ParsedArgs:
        """使用解析器链解析配置。

        Args:
            vllm_config: vLLM 配置

        Returns:
            解析后的参数
        """
        args = ParsedArgs()
        for parser in self.parsers:
            args = parser.parse(args, vllm_config)
        return args


_COMPONENT_METRICS_REGISTRY: dict[str, type["ComponentMetrics"]] = {}


class ComponentMetrics(BaseModel, ABC):
    """组件指标基类。

    每个具体的 ComponentMetrics 类关联于：
    - 指标推导所需的字段（通过 pydantic 模型指定/验证）
    - 将 VllmConfig 解析为字段的解析器
    - 为给定执行上下文推导 FLOPs/字节数的指标方法
    """

    @classmethod
    @abstractmethod
    def component_type(cls) -> str:
        """返回组件类型字符串。

        Returns:
            组件类型标识符
        """
        ...

    @classmethod
    @abstractmethod
    def get_parser(cls) -> ParserChain:
        """返回为所有必需字段提供值的 ParserChain。

        返回的解析器链必须为 ParsedArgs 填充此 ComponentMetrics 类
        定义的每个字段的值。缺失字段会在 from_vllm_config() 调用时
        导致 ValidationError。

        有关哪些解析器提供哪些参数的说明，请参阅各个解析器文档，
        以及 ComponentMetrics 子类字段注释。

        Returns:
            解析器链
        """
        ...

    def __init_subclass__(cls):
        _COMPONENT_METRICS_REGISTRY[cls.component_type()] = cls

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> Self:
        """从 VllmConfig 实例化此组件指标。

        Args:
            vllm_config: vLLM 配置

        Returns:
            组件指标实例

        Raises:
            ValidationError: 如果解析失败
        """
        parser = cls.get_parser()
        parsed_args = parser.parse(vllm_config)
        try:
            return cls.model_validate(parsed_args.model_dump())
        except ValidationError as e:
            raise InvalidComponent(f"Invalid {cls.component_type()} config: {e}") from e

    @classmethod
    def registered_metrics(cls) -> Iterable[type["ComponentMetrics"]]:
        """返回所有已注册的组件指标类。

        Returns:
            组件指标类迭代器
        """
        return iter(_COMPONENT_METRICS_REGISTRY.values())

    @abstractmethod
    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取 FLOPs 细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            FLOPs 细分字典
        """
        ...

    @abstractmethod
    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取读取字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            读取字节数细分字典
        """
        ...

    @abstractmethod
    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取写入字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            写入字节数细分字典
        """
        ...

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取总 FLOPs。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总 FLOPs
        """
        return sum(self.get_num_flops_breakdown(ctx, per_gpu).values())

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取总读取字节数。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总读取字节数
        """
        return sum(self.get_read_bytes_breakdown(ctx, per_gpu).values())

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取总写入字节数。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总写入字节数
        """
        return sum(self.get_write_bytes_breakdown(ctx, per_gpu).values())


#### 解析器 ####


class BaseConfigParser(Parser):
    """基础配置解析器。

    解析模型基础配置。
    提供：vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
         weight_byte_size, activation_byte_size, dp_size, tp_size, pp_size, enable_ep
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析基础配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        model_config = vllm_config.model_config

        args.vocab_size = model_config.get_vocab_size()
        args.hidden_size = model_config.get_hidden_size()
        # 注意：model_config.get_attention_heads() 除以 TP
        # 所以我们直接访问字段以获取总 num_heads
        args.num_attention_heads = get_required(
            model_config.hf_text_config, "num_attention_heads"
        )
        args.num_hidden_layers = get_required(
            model_config.hf_text_config, "num_hidden_layers"
        )

        model_dtype = vllm_config.model_config.dtype

        if isinstance(model_dtype, torch.dtype):
            torch_dtype = model_dtype
        elif isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
        else:
            # FIXME: 更好地处理此情况
            logger.warning(
                "Unknown model_dtype %s, defaulting to bfloat16",
                model_dtype,
            )
            torch_dtype = torch.bfloat16

        args.weight_byte_size = get_dtype_size(torch_dtype)

        # FIXME: 更好地处理此情况，解析激活是否使用 bf16、fp32 等
        args.activation_byte_size = 2

        args.dp_size = vllm_config.parallel_config.data_parallel_size
        args.tp_size = vllm_config.parallel_config.tensor_parallel_size
        args.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel

        return args


#### Attention ####


class BaseAttentionConfigParser(Parser):
    """注意力基础配置解析器。

    解析注意力特定配置。
    提供：num_key_value_heads, head_dim, cache_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析注意力配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        model_config = vllm_config.model_config

        args.num_key_value_heads = model_config.get_total_num_kv_heads()
        args.head_dim = model_config.get_head_size()

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype

        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class AttentionQuantizationConfigParser(Parser):
    """注意力层量化配置解析器。

    解析注意力层的量化配置。
    覆盖：weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析注意力量化配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数

        Raises:
            InvalidComponent: 如果不支持该量化方法
        """
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for attention metrics: {quant_method}"
            )

        return args


class AttentionMetrics(ComponentMetrics):
    """注意力层指标。

    计算注意力层的 FLOPs 和内存访问量。

    Attributes:
        num_hidden_layers: 隐藏层数量（来自 BaseConfigParser）
        hidden_size: 隐藏层大小（来自 BaseConfigParser）
        num_attention_heads: 注意力头数量（来自 BaseConfigParser）
        activation_byte_size: 激活字节大小（来自 BaseConfigParser）
        tp_size: 张量并行大小（来自 BaseConfigParser）
        pp_size: 流水线并行大小（来自 BaseConfigParser）
        num_key_value_heads: KV 头数量（来自 BaseAttentionConfigParser）
        head_dim: 头维度（来自 BaseAttentionConfigParser）
        cache_byte_size: 缓存字节大小（来自 BaseAttentionConfigParser）
        weight_byte_size: 权重字节大小（可被 AttentionQuantizationConfigParser 覆盖）
    """
    # 来自 BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # 来自 BaseAttentionConfigParser
    num_key_value_heads: int = Field(..., gt=0)
    head_dim: int = Field(..., gt=0)
    cache_byte_size: int = Field(..., gt=0)

    # 来自 BaseConfigParser，可被 AttentionQuantizationConfigParser 覆盖
    weight_byte_size: int | float = Field(..., gt=0)

    # TODO: 区分不同注意力层类型（如 SWA、MLA 等）的情况

    @classmethod
    def component_type(cls) -> str:
        return "attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            BaseAttentionConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算注意力层的 FLOPs 细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            FLOPs 细分字典，包含 qkv_proj、attn_qk、attn_av、out_proj
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            # 张量并行沿头维度
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_proj": 2 * T * D * (q + 2 * kv) * d * L,
            "attn_qk": 2 * q * TC * d * L,
            "attn_av": 2 * q * TC * d * L,
            "out_proj": 2 * T * D * q * d * L,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算注意力层的读取字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            读取字节数细分字典
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # 张量并行沿头维度
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        read_bytes = {}

        read_bytes["qkv_input"] = T * D * self.activation_byte_size * L
        read_bytes["qkv_weight"] = int(D * (q + 2 * kv) * d * self.weight_byte_size * L)

        # 注意力输入读取决于预填充和解码阶段
        # 预填充：读取 Q、K、V 激活（都使用 activation_byte_size）
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d
                * self.activation_byte_size
                * L
            )

        # 解码：读取 Q 激活 + 从缓存读取 K、V（使用 cache_byte_size）
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * d * self.activation_byte_size * L
                + 2 * ctx.decode_context_len * kv * d * self.cache_byte_size * L
            )

        read_bytes["out_input"] = T * q * d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算注意力层的写入字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            写入字节数细分字典
        """
        L, D, q, kv, d = (
            self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # 张量并行沿头维度
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_output": T * (q + 2 * kv) * d * self.activation_byte_size * L,
            "kv_cache": 2 * T * kv * d * self.cache_byte_size * L,
            "out_output": T * D * self.activation_byte_size * L,
        }


#### FFN ####


class BaseFfnConfigParser(Parser):
    """FFN 和 MoE 配置解析器。

    解析 FFN 和 MoE 配置。
    提供：intermediate_size, num_experts, num_experts_per_tok,
         moe_intermediate_size, num_shared_experts, num_moe_layers
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析 FFN 和 MoE 配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        args.intermediate_size = getattr(cfg, "intermediate_size", args.hidden_size * 4)

        # 尝试不同的命名约定
        args.num_experts = vllm_config.model_config.get_num_experts()
        args.num_experts_per_tok = getattr_from_list(
            cfg, ["num_experts_per_tok", "moe_topk"], 0
        )
        args.moe_intermediate_size = getattr_from_list(
            cfg, ["moe_intermediate_size", "intermediate_size"], 0
        )
        args.num_shared_experts = getattr_from_list(
            cfg, ["n_shared_experts", "num_shared_experts"], 0
        )

        is_moe = args.num_experts != 0
        # 假设所有 MoE 层
        args.num_moe_layers = args.num_hidden_layers if is_moe else 0

        return args


class FfnParallelParser(Parser):
    """FFN 并行配置解析器。

    解析 FFN 并行配置。
    提供：ffn_tp_size, ffn_ep_size

    注意：ffn_tp_size 不直接等于 tp_size 参数。
    例如：如果使用 DP2TP4，FFN 将使用 TP8（或如果启用 EP 则为 EP8）。
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析 FFN 并行配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        # 注意：ffn tp_size 不直接等于 tp_size 参数
        # 例如：如果使用 DP2TP4，FFN 将使用 TP8（或如果启用 EP 则为 EP8）
        if args.enable_ep:
            ffn_tp_size, ffn_ep_size = 1, args.dp_size * args.tp_size
        else:
            ffn_tp_size, ffn_ep_size = args.dp_size * args.tp_size, 1

        args.ffn_tp_size = ffn_tp_size
        args.ffn_ep_size = ffn_ep_size

        return args


class InterleaveMoeLayerStepParser(Parser):
    """交错 MoE 层步长解析器。

    解析 Llama4 等模型的 interleave_moe_layer_step 字段。
    覆盖：num_moe_layers
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析交错 MoE 层配置。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if (
            hasattr(cfg, "interleave_moe_layer_step")
            and cfg.interleave_moe_layer_step > 0
        ):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if (layer + 1) % cfg.interleave_moe_layer_step == 0
                ]
            )

        return args


class MoeLayerFreqParser(Parser):
    """MoE 层频率解析器。

    解析 Deepseek 等模型的 moe_layer_freq 和 first_k_dense_replace 字段。
    覆盖：num_moe_layers
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析 MoE 层频率配置。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数
        """
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if hasattr(cfg, "moe_layer_freq") and hasattr(cfg, "first_k_dense_replace"):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if layer >= cfg.first_k_dense_replace
                    and layer % cfg.moe_layer_freq == 0
                ]
            )

        return args


class FfnQuantizationConfigParser(Parser):
    """FFN 层量化配置解析器。

    解析 FFN 层的量化配置。
    覆盖：weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """解析 FFN 量化配置参数。

        Args:
            args: 当前解析参数
            vllm_config: vLLM 配置

        Returns:
            更新后的解析参数

        Raises:
            InvalidComponent: 如果不支持该量化方法
        """
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for FFN metrics: {quant_method}"
            )

        return args


class FfnMetrics(ComponentMetrics):
    """前馈网络层指标。

    计算 FFN 层（包括 Dense FFN 和 MoE）的 FLOPs 和内存访问量。

    Attributes:
        num_hidden_layers: 隐藏层数量
        hidden_size: 隐藏层大小
        activation_byte_size: 激活字节大小
        pp_size: 流水线并行大小
        ffn_tp_size: FFN 张量并行大小
        ffn_ep_size: FFN 专家并行大小
        intermediate_size: 中间层大小
        num_experts: 专家数量
        num_experts_per_tok: 每个 token 激活的专家数
        moe_intermediate_size: MoE 中间层大小
        num_shared_experts: 共享专家数量
        num_moe_layers: MoE 层数量
        weight_byte_size: 权重字节大小
    """
    # 来自 BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # 来自 FfnParallelParser
    ffn_tp_size: int = Field(..., gt=0)
    ffn_ep_size: int = Field(..., gt=0)

    # 来自 BaseFfnConfigParser
    intermediate_size: int = Field(..., gt=0)
    num_experts: int = Field(0)
    num_experts_per_tok: int = Field(1)
    moe_intermediate_size: int = Field(0)
    num_shared_experts: int = Field(0)

    # 来自 BaseConfigParser，可被 InterleaveMoeLayerStep 或 MoeLayerFreq 覆盖
    num_moe_layers: int = Field(..., ge=0)

    # FIXME: 可能需要更细粒度（dense_weight_byte_size、moe_routed_weight_byte_size、
    # moe_shared_weight_byte_size），因为它可能与其他组件（如 attn）的字节大小不同，
    # 甚至彼此不同。

    # 来自 BaseConfigParser，可被 FfnQuantizationConfigParser 覆盖
    weight_byte_size: int | float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_moe_fields(self) -> Self:
        """验证 MoE 相关字段在 num_moe_layers > 0 时是否正确设置。

        Returns:
            self

        Raises:
            AssertionError: 如果 MoE 字段设置不正确
        """
        if self.num_moe_layers > 0:
            assert self.num_experts, f"{self.num_experts=}"
            assert self.num_experts_per_tok, f"{self.num_experts_per_tok=}"
            assert self.moe_intermediate_size, f"{self.moe_intermediate_size=}"
        return self

    @classmethod
    def component_type(cls) -> str:
        return "ffn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            FfnParallelParser(),
            BaseFfnConfigParser(),
            InterleaveMoeLayerStepParser(),
            MoeLayerFreqParser(),
            FfnQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算 FFN 层的 FLOPs 细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            FLOPs 细分字典，包含 dense_ffn、routed_ffn、shared_ffn
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        flops = {}

        # Dense FFN 层（SwiGLU：3 个线性层：up、gate、down）
        if Ld:
            flops["dense_ffn"] = 2 * D * 3 * DI * T * Ld

        # MoE 路由专家（每个 token 激活 E 个专家）
        if Lm and E:
            flops["routed_ffn"] = 2 * D * 3 * MI * num_activated_tokens * Lm

        # MoE 共享专家（所有 S 个共享专家为每个 token 运行）
        if Lm and S:
            flops["shared_ffn"] = 2 * D * 3 * MI * S * T * Lm

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算 FFN 层的读取字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            读取字节数细分字典
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()
        num_experts = self.num_experts

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size
            if num_experts is not None:
                num_experts //= self.ffn_ep_size

        read_bytes = {}

        # Dense FFN 层（3 个 GEMM：up、gate、down 投影 + SiLU 激活）
        if Ld:
            read_bytes["dense_up_gate_input"] = int(
                T * D * self.activation_byte_size * Ld
            )
            read_bytes["dense_up_gate_weights"] = int(
                2 * D * DI * self.weight_byte_size * Ld
            )
            read_bytes["dense_silu_input"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_input"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_weights"] = int(D * DI * self.weight_byte_size * Ld)

        if Lm:
            # MoE 路由专家读取
            if E:
                # FIXME: 暂时假设完美负载均衡
                num_activated_experts = min(num_activated_tokens, num_experts)

                read_bytes["routed_up_gate_input"] = int(
                    num_activated_tokens * D * self.activation_byte_size * Lm
                )
                read_bytes["routed_up_gate_weights"] = int(
                    2 * D * MI * num_activated_experts * self.weight_byte_size * Lm
                )
                read_bytes["routed_silu_input"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_input"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_weights"] = int(
                    D * MI * num_activated_experts * self.weight_byte_size * Lm
                )

            # MoE 共享专家读取
            if S:
                read_bytes["shared_up_gate_input"] = int(
                    T * D * self.activation_byte_size * Lm
                )
                read_bytes["shared_up_gate_weights"] = int(
                    2 * D * MI * S * self.weight_byte_size * Lm
                )
                read_bytes["shared_silu_input"] = int(
                    2 * T * MI * S * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_input"] = int(
                    T * MI * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_weights"] = int(
                    D * MI * S * self.weight_byte_size * Lm
                )

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算 FFN 层的写入字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            写入字节数细分字典
        """
        L, D, DI = self.num_hidden_layers, self.hidden_size, self.intermediate_size
        Lm, E, MI, S = (
            self.num_moe_layers,
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        Ld = L - Lm

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        write_bytes = {}

        # Dense FFN 层
        if Ld:
            write_bytes["dense_up_gate_output"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_silu_output"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_down_output"] = int(
                T * D * self.activation_byte_size * Ld
            )

        # MoE 输出
        if Lm:
            if E:
                write_bytes["routed_up_gate_output"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_silu_output"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_down_output"] = int(
                    num_activated_tokens * D * self.activation_byte_size * Lm
                )
            if S:
                write_bytes["shared_up_gate_output"] = int(
                    2 * T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_silu_output"] = int(
                    T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_down_output"] = int(
                    T * S * D * self.activation_byte_size * Lm
                )

        return write_bytes


#### Unembed ####


class UnembedMetrics(ComponentMetrics):
    """输出层（Unembed）指标。

    计算输出层（从隐藏状态到词汇表 logits）的 FLOPs 和内存访问量。

    Attributes:
        hidden_size: 隐藏层大小
        vocab_size: 词汇表大小
        weight_byte_size: 权重字节大小
        activation_byte_size: 激活字节大小
        tp_size: 张量并行大小
    """
    # 来自 BaseConfigParser
    hidden_size: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    weight_byte_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)

    tp_size: int

    @classmethod
    def component_type(cls) -> str:
        return "unembed"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算输出层的 FLOPs 细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            FLOPs 细分字典，包含 unembed
        """
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "unembed": 2 * T * D * V,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算输出层的读取字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            读取字节数细分字典
        """
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "input": T * D * self.activation_byte_size,
            "weight": D * V * self.weight_byte_size,
        }

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """计算输出层的写入字节数细分。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            写入字节数细分字典
        """
        V = self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "output": T * V * self.activation_byte_size,
        }


#### ModelMetrics ####


class ModelMetrics:
    """模型整体指标。

    解析 vllm_config 来实例化每个组件的指标。
    如果没有实例化任何组件指标，is_enabled() 将返回 False。

    Attributes:
        vllm_config: vLLM 配置
        metrics: 组件指标列表
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        """初始化模型指标。

        Args:
            vllm_config: vLLM 配置
        """
        self.vllm_config = vllm_config

        self.metrics: list[ComponentMetrics] = []
        for metric_cls in ComponentMetrics.registered_metrics():
            try:
                metric = metric_cls.from_vllm_config(vllm_config)
                self.metrics.append(metric)
                logger.info(
                    "Instantiated ComponentMetrics [%s] with (%s)",
                    metric.component_type(),
                    str(metric),
                )
            except InvalidComponent as e:
                logger.debug(
                    "Failed to instantiate %s from %s",
                    metric_cls.component_type(),
                    str(e),
                )

    def is_enabled(self) -> bool:
        """检查指标是否已启用。

        Returns:
            是否启用了至少一个组件指标
        """
        return len(self.metrics) > 0

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取模型总 FLOPs。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总 FLOPs
        """
        return sum(metric.get_num_flops(ctx, per_gpu) for metric in self.metrics)

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取模型总读取字节数。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总读取字节数
        """
        return sum(metric.get_read_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        """获取模型总写入字节数。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            总写入字节数
        """
        return sum(metric.get_write_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取模型 FLOPs 细分（按组件前缀）。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            FLOPs 细分字典
        """
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_num_flops_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取模型读取字节数细分（按组件前缀）。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            读取字节数细分字典
        """
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_read_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """获取模型写入字节数细分（按组件前缀）。

        Args:
            ctx: 执行上下文
            per_gpu: 是否返回每 GPU 数据

        Returns:
            写入字节数细分字典
        """
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_write_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_step_perf_stats_per_gpu(
        self, scheduler_output: SchedulerOutput
    ) -> PerfStats:
        """根据调度的 token 计算当前步骤的性能统计。

        Args:
            scheduler_output: 调度器输出

        Returns:
            每 GPU 性能统计
        """
        t0 = time.monotonic()

        # 构建单一批次上下文
        ctx = ExecutionContext()

        # 处理新请求（这些处于预填充阶段）
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # 对于新请求，context_len = num_computed_tokens + num_tokens
            # num_computed_tokens 表示序列中先前计算的 token
            context_len = new_req.num_computed_tokens + num_tokens
            ctx.add(num_tokens, context_len, is_prefill=True)

        # 处理缓存请求（继续的请求）
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # 对于缓存请求，我们有当前的 num_computed_tokens
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            context_len = num_computed_tokens + num_tokens

            # 缓存请求通常处于解码阶段（num_tokens == 1）
            # 除非它们进行 chunked prefill（num_tokens > 1）
            is_prefill = num_tokens > 1
            ctx.add(num_tokens, context_len, is_prefill)

        num_flops_breakdown = self.get_num_flops_breakdown(ctx, True)
        read_bytes_breakdown = self.get_read_bytes_breakdown(ctx, True)
        write_bytes_breakdown = self.get_write_bytes_breakdown(ctx, True)
        perf_stats = PerfStats(
            sum(num_flops_breakdown.values()),
            sum(read_bytes_breakdown.values()),
            sum(write_bytes_breakdown.values()),
        )

        if envs.VLLM_DEBUG_MFU_METRICS:
            perf_stats.debug_stats = DebugPerfStats(
                time.monotonic() - t0,
                ctx.num_prefill_requests,
                ctx.num_decode_requests,
                asdict(ctx),
                num_flops_breakdown,
                read_bytes_breakdown,
                write_bytes_breakdown,
            )

        return perf_stats


#### 日志记录 ####


class PerfMetricsDebugLogging:
    """性能指标调试日志记录。

    累积多个批次的调试统计信息，
    用于详细分析 MFU 计算。

    Attributes:
        total_calc_duration: 总计算时长
        total_num_prefill_requests: 总预填充请求数
        total_num_decode_requests: 总解码请求数
        total_num_batches: 总批次数
        total_context_breakdown: 总上下文细分
        total_num_flops_per_gpu_breakdown: 总 FLOPs 细分
        total_read_bytes_per_gpu_breakdown: 总读取字节数细分
        total_write_bytes_per_gpu_breakdown: 总写入字节数细分
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有累积统计。"""
        self.total_calc_duration: float = 0.0
        self.total_num_prefill_requests: int = 0
        self.total_num_decode_requests: int = 0
        self.total_num_batches: int = 0
        self.total_context_breakdown: dict[str, int] = {}
        self.total_num_flops_per_gpu_breakdown: dict[str, int] = {}
        self.total_read_bytes_per_gpu_breakdown: dict[str, int] = {}
        self.total_write_bytes_per_gpu_breakdown: dict[str, int] = {}

    def observe(self, debug_stats: DebugPerfStats) -> None:
        """累积调试统计。

        Args:
            debug_stats: 单次迭代的调试统计
        """
        self.total_calc_duration += debug_stats.calc_duration
        self.total_num_prefill_requests += debug_stats.num_prefill_requests
        self.total_num_decode_requests += debug_stats.num_decode_requests
        self.total_num_batches += 1

        for dst, src in zip(
            [
                self.total_context_breakdown,
                self.total_num_flops_per_gpu_breakdown,
                self.total_read_bytes_per_gpu_breakdown,
                self.total_write_bytes_per_gpu_breakdown,
            ],
            [
                debug_stats.context_breakdown,
                debug_stats.num_flops_per_gpu_breakdown,
                debug_stats.num_read_bytes_per_gpu_breakdown,
                debug_stats.num_write_bytes_per_gpu_breakdown,
            ],
        ):
            assert isinstance(src, dict)
            for key, val in src.items():
                dst[key] = dst.get(key, 0) + val

    def log(self, log_fn, log_prefix: str, delta_time: float):
        """记录累积的调试统计。

        Args:
            log_fn: 日志函数
            log_prefix: 日志前缀
            delta_time: 时间间隔（秒）
        """
        # 美化输出细分
        total_num_flops_per_gpu_breakdown = {
            k: f"{v / 1e12:.1f}TF"
            for k, v in self.total_num_flops_per_gpu_breakdown.items()
        }
        total_read_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_read_bytes_per_gpu_breakdown.items()
        }
        total_write_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_write_bytes_per_gpu_breakdown.items()
        }

        logger.debug(
            "%sMFU details: %s",
            log_prefix,
            json.dumps(
                {
                    "prefill_reqs": self.total_num_prefill_requests,
                    "decode_reqs": self.total_num_decode_requests,
                    "num_batches": self.total_num_batches,
                    "context_breakdown": self.total_context_breakdown,
                    "flops_breakdown": total_num_flops_per_gpu_breakdown,
                    "num_read_bytes_breakdown": total_read_bytes_per_gpu_breakdown,
                    "num_write_bytes_breakdown": (total_write_bytes_per_gpu_breakdown),
                    "duration": f"{delta_time:.1f}s",
                    "mfu_calc_overhead": (
                        f"{self.total_calc_duration / delta_time:.1%}"
                    ),
                },
                indent=2,
            ),
        )


class PerfMetricsLogging:
    """性能指标日志记录。

    累积和记录性能指标，如 MFU（Model Flops Utilization）。

    Attributes:
        vllm_config: vLLM 配置
        pp_size: 流水线并行大小
        debug_logging: 调试日志记录（可选）
        last_log_time: 上次日志记录时间
        total_num_flops_per_gpu: 累积每 GPU FLOPs
        total_read_bytes_per_gpu: 累积每 GPU 读取字节数
        total_write_bytes_per_gpu: 累积每 GPU 写入字节数
    """

    def __init__(self, vllm_config: VllmConfig):
        """初始化性能指标日志记录。

        Args:
            vllm_config: vLLM 配置
        """
        self.vllm_config = vllm_config
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size

        self.debug_logging: PerfMetricsDebugLogging | None = None
        if envs.VLLM_DEBUG_MFU_METRICS:
            self.debug_logging = PerfMetricsDebugLogging()

        self.reset()

    def reset(self):
        """重置累积的性能指标。"""
        self.last_log_time = time.monotonic()

        self.total_num_flops_per_gpu: int = 0
        self.total_read_bytes_per_gpu: int = 0
        self.total_write_bytes_per_gpu: int = 0

        if self.debug_logging:
            self.debug_logging.reset()

    def observe(self, perf_stats: PerfStats) -> None:
        """累积性能统计。

        Args:
            perf_stats: 性能统计
        """
        self.total_num_flops_per_gpu += perf_stats.num_flops_per_gpu
        self.total_read_bytes_per_gpu += perf_stats.num_read_bytes_per_gpu
        self.total_write_bytes_per_gpu += perf_stats.num_write_bytes_per_gpu

        if self.debug_logging:
            assert perf_stats.debug_stats is not None
            self.debug_logging.observe(perf_stats.debug_stats)

    def log(self, log_fn=logger.info, log_prefix: str = "") -> None:
        """记录累积的性能指标。

        计算并记录平均 TFLOPS 和内存带宽（GB/s）。

        Args:
            log_fn: 日志函数
            log_prefix: 日志前缀
        """
        if not (
            self.total_num_flops_per_gpu
            or self.total_read_bytes_per_gpu
            or self.total_write_bytes_per_gpu
        ):
            return

        now = time.monotonic()
        delta_time = now - self.last_log_time

        if delta_time <= 0.0:
            avg_tflops_per_gpu = 0.0
            avg_gbps_per_gpu = 0.0
        else:
            avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
            avg_gbps_per_gpu = (
                (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
                / delta_time
                / 1e9
            )

        log_fn(
            "%sMFU: %.1f TF/s/GPU %.1f GB/s/GPU",
            log_prefix,
            avg_tflops_per_gpu,
            avg_gbps_per_gpu,
        )

        if self.debug_logging:
            self.debug_logging.log(log_fn, log_prefix, delta_time)

        self.reset()


#### Prometheus 集成 ####


class PerfMetricsProm:
    """Prometheus 性能指标记录器。

    在 Prometheus 中记录性能指标。

    平均 TFLOPS（每秒万亿次浮点运算）可以使用 PromQL 查询计算：

      rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

    平均内存带宽（GB/s）可以使用以下查询计算：

      (rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
       rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9

    Attributes:
        counter_flops: FLOPs 计数器
        counter_read_bytes: 读取字节数计数器
        counter_write_bytes: 写入字节数计数器
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        """初始化 Prometheus 性能指标。

        Args:
            vllm_config: vLLM 配置
            labelnames: 标签名称列表
            per_engine_labelvalues: 每引擎标签值映射
        """
        counter_flops = self._counter_cls(
            name="vllm:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_flops = create_metric_per_engine(
            counter_flops, per_engine_labelvalues
        )

        counter_read_bytes = self._counter_cls(
            name="vllm:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_read_bytes = create_metric_per_engine(
            counter_read_bytes, per_engine_labelvalues
        )

        counter_write_bytes = self._counter_cls(
            name="vllm:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_write_bytes = create_metric_per_engine(
            counter_write_bytes, per_engine_labelvalues
        )

    def observe(self, perf_stats: PerfStats, engine_idx: int = 0):
        """记录性能统计到 Prometheus。

        Args:
            perf_stats: 性能统计
            engine_idx: 引擎索引
        """
        if not (
            perf_stats.num_flops_per_gpu
            or perf_stats.num_read_bytes_per_gpu
            or perf_stats.num_write_bytes_per_gpu
        ):
            return
        self.counter_flops[engine_idx].inc(perf_stats.num_flops_per_gpu)
        self.counter_read_bytes[engine_idx].inc(perf_stats.num_read_bytes_per_gpu)
        self.counter_write_bytes[engine_idx].inc(perf_stats.num_write_bytes_per_gpu)


## 工具函数 ##


def get_required(obj: object, attr: str):
    """从对象获取属性，如果不存在则抛出 InvalidComponentError。

    Args:
        obj: 对象
        attr: 属性名

    Returns:
        属性值

    Raises:
        InvalidComponent: 如果属性不存在
    """
    if not hasattr(obj, attr):
        raise InvalidComponent(f"Missing required attr {attr} in config")
    return getattr(obj, attr)


def getattr_from_list(obj: object, attrs: list[str], default: object = None):
    """尝试从对象获取列表中第一个存在的属性。

    Args:
        obj: 对象
        attrs: 属性名列表
        default: 默认值

    Returns:
        属性值或默认值
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default
