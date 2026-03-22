# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-K 和 Top-P 采样器模块。

本模块实现了 top-k 和 top-p 滤波采样功能，负责：
- 应用 top-k 和 top-p 约束到 logits
- 执行加权随机采样
- 支持多种后端实现（原生、FlashInfer、CPU、ROCm aiter）
- 优化采样性能

主要类：
- TopKTopPSampler: Top-K 和 Top-P 采样器
"""


import torch
import torch.nn as nn
from packaging import version

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

logger = init_logger(__name__)


class TopKTopPSampler(nn.Module):
    """执行可选的 top-k 和 top-p 滤波，然后进行加权随机采样的模块。

    实现可能会就地更新 logits 张量。

    支持多种后端：
    - 原生 PyTorch 实现（通用）
    - FlashInfer（CUDA，需要显式启用）
    - CPU 优化实现（x86 架构）
    - ROCm aiter（AMD GPU）

    Attributes:
        logprobs_mode: logprobs 模式配置
    """

    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
        """初始化 TopKTopPSampler。

        Args:
            logprobs_mode: logprobs 模式，默认为 "raw_logprobs"
        """
        super().__init__()
        self.logprobs_mode = logprobs_mode
        # 如果需要返回中间 logprobs/logits（在 top_k/top_p 之后），
        # 则不使用 flashinfer 优化
        if (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and current_platform.is_cuda()
        ):
            if envs.VLLM_USE_FLASHINFER_SAMPLER:
                from vllm.v1.attention.backends.flashinfer import FlashInferBackend

                capability = current_platform.get_device_capability()
                assert capability is not None
                if not FlashInferBackend.supports_compute_capability(capability):
                    capability_str = capability.as_version_str()
                    raise RuntimeError(
                        f"FlashInfer 不支持计算能力 {capability_str}，"
                        f"请取消设置 VLLM_USE_FLASHINFER_SAMPLER=1。"
                    )
                # 用户必须通过 VLLM_USE_FLASHINFER_SAMPLER=1 显式启用
                logger.info_once("使用 FlashInfer 进行 top-p 和 top-k 采样。", scope="global")
                self.forward = self.forward_cuda
            else:
                logger.debug_once(
                    "FlashInfer top-p/top-k 采样可用但默认禁用。"
                    "在为您的工作负载验证准确性后，设置 VLLM_USE_FLASHINFER_SAMPLER=1 以启用。"
                )
                self.forward = self.forward_native

        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            # 对 POWERPC 和 RISCV 回退到原生实现。
            # 在 PowerPC 上，argmax 与 torch.compile 一起使用时会产生不正确的输出。
            # PR: https://github.com/vllm-project/vllm/pull/26987
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif (
            logprobs_mode not in ("processed_logits", "processed_logprobs")
            and rocm_aiter_ops.is_enabled()
        ):
            try:
                import aiter.ops.sampling  # noqa: F401

                self.aiter_ops = torch.ops.aiter
                logger.info_once("在 ROCm 上使用 aiter 采样器（延迟导入，仅采样）。")
                self.forward = self.forward_hip
            except ImportError:
                logger.warning_once(
                    "aiter.ops.sampling 在 ROCm 上不可用。回退到 forward_native 实现。"
                )
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Top-k 和 top-p 采样的 PyTorch 原生实现。

        logits 张量可能会被就地更新。

        Args:
            logits: 输入的 logits
            generators: 随机数生成器字典
            k: top-k 张量
            p: top-p 张量

        Returns:
            (采样 token IDs, 处理后的 logits/logprobs 或 None)
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators), logits_to_return

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """更优化的 top-k 和 top-p 采样 CUDA 实现。

        Args:
            logits: 输入的 logits
            generators: 随机数生成器字典
            k: top-k 张量
            p: top-p 张量

        Returns:
            (采样 token IDs, 处理后的 logits/logprobs 或 None)
        """
        # 当不需要排序时，我们更喜欢 `random_sample` 而不是 `flashinfer_sample`。
        # 这是因为 `random_sample` 不需要 CPU-GPU 同步，而 `flashinfer_sample` 需要。
        if (k is None and p is None) or generators:
            if generators:
                logger.debug_once(
                    "FlashInfer 0.2.3+ 不支持每请求生成器。回退到 PyTorch 原生实现。"
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ("processed_logits", "processed_logprobs"), (
            "FlashInfer 不支持返回 logits/logprobs"
        )
        # flashinfer 采样函数需要连续的 logits。
        # 在 flex_attn/triton_attn fp32 推理中，logits 可能不连续，
        # 因为 logits_processor 中的切片操作。
        return flashinfer_sample(logits.contiguous(), k, p, generators), None

    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """CPU 的 PyTorch 原生 top-k 和 top-p 采样实现。

        logits 张量可能会被就地更新。

        Args:
            logits: 输入的 logits
            generators: 随机数生成器字典
            k: top-k 张量
            p: top-p 张量

        Returns:
            (采样 token IDs, 处理后的 logits/logprobs 或 None)
        """
        logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        if len(generators) != logits.shape[0]:
            return compiled_random_sample(logits), logits_to_return

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

        return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return

    def forward_hip(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ROCm/aiter 优化路径（结构与 forward_cuda 相同）。

        Args:
            logits: 输入的 logits
            generators: 随机数生成器字典
            k: top-k 张量
            p: top-p 张量

        Returns:
            (采样 token IDs, 处理后的 logits/logprobs 或 None)
        """
        # FIXME: 修复 aiter_sampler 的精度问题并移除此标志
        DISABLE_AITER_SAMPLER = True
        if (k is None and p is None) or generators:
            if generators:
                logger.warning_once(
                    "aiter 采样器不支持每请求生成器；回退到 PyTorch 原生实现。"
                )
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in (
            "processed_logits",
            "processed_logprobs",
        ), "aiter 采样器不支持返回 logits/logprobs。"
        if DISABLE_AITER_SAMPLER:
            return self.forward_native(logits, generators, k, p)
        return self.aiter_sample(logits, k, p, generators), None

    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """使用 aiter 操作从 logits 采样。

        Args:
            logits: 输入的 logits
            k: top-k 张量
            p: top-p 张量
            generators: 随机数生成器字典

        Returns:
            采样 token IDs

        Raises:
            RuntimeError: 当没有激活的 top-k 或 top-p 时
        """
        use_top_k = k is not None
        use_top_p = p is not None
        # 联合 k+p 路径
        if use_top_p and use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(
                probs,
                None,
                *_to_tensor_scalar_tuple(k),
                *_to_tensor_scalar_tuple(p),
                deterministic=True,
            )
            return next_token_ids.view(-1)
        # 仅 Top-p 路径
        elif use_top_p:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_p_sampling_from_probs(
                probs, None, *_to_tensor_scalar_tuple(p), deterministic=True
            )
            return next_token_ids.view(-1)
        # 仅 Top-k 路径
        elif use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            renorm_probs = self.aiter_ops.top_k_renorm_probs(
                probs, *_to_tensor_scalar_tuple(k)
            )
            return torch.multinomial(renorm_probs, num_samples=1).view(-1)
        raise RuntimeError("aiter_sample 被调用时没有激活的 top-k 或 top-p。")


# 注意：这是 https://github.com/pytorch/pytorch/pull/151218 的临时解决方案
@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    """编译的随机采样函数。

    Args:
        logits: 输入的 logits

    Returns:
        采样 token IDs
    """
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    """应用 top-k 和 top-p 约束。

    Args:
        logits: 输入的 logits
        k: top-k 张量
        p: top-p 张量

    Returns:
        处理后的 logits
    """
    if p is None and k is None:
        return logits

    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # 对小批次大小使用 PyTorch 排序实现。
    return apply_top_k_top_p_pytorch(logits, k, p)


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """应用 top-k 和 top-p 掩码到 logits。

    如果使用 top-p，此函数将对 logits 张量进行排序，
    这对于大批次可能较慢。

    logits 张量可能会被就地更新。

    Args:
        logits: 输入的 logits
        k: top-k 张量
        p: top-p 张量
        allow_cpu_sync: 是否允许 CPU 同步

    Returns:
        处理后的 logits
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # 仅 top-k 情况下避免对 vocab 进行排序。
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # 应用 top-k。
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # 形状：B
        # 获取所有 top-k 值。
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # 应用 top-p。
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # 至少保留一个
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # 重新排序概率。
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """应用 top-k 掩码到 logits。

    此实现不涉及对整个 vocab 进行排序。
    但请注意，它涉及 GPU->CPU 同步，这可能对异步调度性能有害。

    logits 张量可能会被就地更新。

    Args:
        logits: 输入的 logits
        k: top-k 张量

    Returns:
        处理后的 logits
    """
    no_top_k_mask = k == logits.shape[1]
    # 将非 top-k 行设置为 1，以便我们可以 gather。
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values 张量形状为 [batch_size, max_top_k]。
    # 将 top k 转换为范围 [0, max_top_k) 内的基于 0 的索引。
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # 处理非 topk 行。
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """从概率中随机采样。

    我们使用此函数而不是 torch.multinomial，因为 torch.multinomial
    会导致 CPU-GPU 同步。

    Args:
        probs: 概率张量
        generators: 随机数生成器字典

    Returns:
        采样 token IDs
    """
    q = torch.empty_like(probs)
    # 注意 (woosuk): 为了批量处理没有自己种子的请求（这是常见情况），
    # 我们首先假设每个请求都没有自己的种子。然后，我们为有自己种子的
    # 请求覆盖值。
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): 这可能很慢，因为我们逐个处理每个请求。优化此问题。
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """使用 FlashInfer 从 logits 采样。

    从统计上讲，此函数等价于 `random_sample` 函数。
    但是，此函数更快，因为它通过拒绝采样避免了 logits 张量排序。

    注意：此函数的输出不一定与 `random_sample` 函数的输出匹配。
    它只保证输出在统计上等价。

    注意：此函数包含 CPU-GPU 同步，而 `random_sample` 不包含。
    在前向传递结束时调用此函数以最小化同步开销。

    Args:
        logits: 输入的 logits
        k: top-k 张量
        p: top-p 张量
        generators: 随机数生成器字典

    Returns:
        采样 token IDs

    Raises:
        ImportError: 当 FlashInfer 版本 < 0.2.3 时
    """
    import flashinfer

    if version.parse(flashinfer.__version__) < version.parse("0.2.3"):
        raise ImportError(
            "Top-k 和 top-p 采样需要 FlashInfer 版本 >= 0.2.3。"
        )

    assert not (k is None and p is None)
    if k is None:
        # 仅 Top-p。
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # 仅 Top-k。
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Top-k 和 top-p 都有。
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)


def _to_tensor_scalar_tuple(x):
    """将参数转换为 (张量，标量) 元组。

    用于 aiter 操作的参数转换。

    Args:
        x: 输入参数（张量或标量）

    Returns:
        (张量，0) 如果 x 是张量，否则 (None, x)
    """
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)
