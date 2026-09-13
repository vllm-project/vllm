# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

# [CN] 文件总览：hidden states -> logits 的最后一层（lm_head 投影 + TP 汇聚）
# [CN] 职责：把模型最后一层的隐向量乘上 lm_head 得到词表 logits，可选做 soft_cap / scale。
# [CN] 链路：模型 forward 末尾 -> LogitsProcessor.forward() -> logits -> sampler.py
# [CN] 三条路径：
# [CN]   forward()           —— 完整 logits（TP 下需跨卡聚合），采样流程的主力
# [CN]   get_top_tokens()    —— 只要 argmax 时的省通信路径（不聚合整份词表）
# [CN]   get_top_k_tokens()  —— 只要 top-k 时的省通信路径（投机解码 verify 用得最多）
# [CN] 关键机制：
# [CN]   1. TP 分片：lm_head 按词表维切分，每张卡只算出自己那一段 vocab 的 logits。
# [CN]   2. org_vocab_size 裁剪：词表常被 pad 到对 TP / 算子友好的倍数，
# [CN]      取 logits 前必须裁掉尾部填充，否则采样可能返回不存在的 token。
# [CN]   3. head_dtype 与 model dtype 解耦：RL 场景要求 fp32 头以保证训推一致，
# [CN]      CUDA/ROCm 上用 torch.mm(out_dtype=fp32) 直接累积，避免每步物化一份 fp32 权重副本。
# [CN] 易错点：省通信路径要求 scale > 0。负的 scale 会翻转 argmax 语义，
# [CN]         这里选择直接报错而不是返回一个看起来合理但错了的结果。

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)


# [CN] 探测一次并永久缓存：flashinfer 的 radix top-k 在整份词表上约比 torch.topk 快一倍。
# [CN] 非 CUDA 或没装 flashinfer 时降级为 torch.topk，日志只打一次（info_once），不刷屏。

@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """FlashInfer's radix top-k, or None for torch.topk.

    The top-k spans the vocabulary, where the radix kernel is about twice
    torch.topk.
    """
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
        logger.info_once(
            "flashinfer is unavailable; vocab-parallel top-k uses torch.topk, "
            "at roughly half the speed."
        )
        return None
    from flashinfer import top_k

    return top_k


# [CN] 统一入口：连同 is_cuda 检查一起兜住 —— impl 存在不代表当前输入能用（可能是 CPU 张量）。

def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    return impl(scores, k, sorted=True, deterministic=True)


# [CN] PluggableLayer.register 让这个 key 可以被替换，即「怎么算 logits」是插拔的而非写死的。

# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    # [CN] logits_as_input=True 时本层退化为「只做 soft_cap/scale 的透传」，不再调用 lm_head ——
    # [CN] 某些多模态 head 会自己算好 logits 再传进来。

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None

    # [CN] 主干：要么接已经算好的 logits，要么自己投影；随后统一做 soft_cap -> scale。
    # [CN] 顺序不可换：soft_cap 要求先 tanh 压缩再放大（这是 Gemma 2 的定义），scale 最后乘。

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(
                hidden_states, lm_head, embedding_bias, skip_gather
            )
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    # [CN] TP 汇聚有两种：all_gather（每卡都拿到全量）与 gather（只有 rank 0 拿结果，其余返回 None）。
    # [CN] TPU/XLA 要求严格 SPMD —— 所有设备必须执行完全相同的一组算子，因此只能用 all_gather。

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    # [CN] hidden states -> logits 投影，并处理 head_dtype 与 model dtype 不一致的情况。
    # [CN] 快路径（两者一致）直接复用 quant_method.apply，完全不关心下面的分支。

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )

        # A quant config that excludes lm_head hands out UnquantizedLinearMethod
        # rather than UnquantizedEmbeddingMethod, so accept both: either way the
        # weight is plain and `lm_head.weight` can be cast directly.
        # [CN] head_dtype 不等于 model dtype 时，lm_head 必须是无量化实现：量化权重无法当场 cast。
        # [CN] 同时接受 Linear / Embedding 两种无量化实现，是因为 quant config 排除 lm_head 时
        # [CN] 给出的方法类型并不统一（有时会下落到 UnquantizedLinearMethod）。

        if not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            self.head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            # Accumulate the projection directly into fp32. This avoids
            # materializing an fp32 copy of the lm_head weight on every step,
            # unlike casting both operands. `torch.mm(out_dtype=...)` only
            # supports fp32 output for fp16/bf16 inputs, and is only
            # implemented for CUDA and ROCm (the latter via the non-Lt GEMM
            # path); other platforms fall back to the cast path below.
            # [CN] fp32 直出路径：torch.mm(out_dtype=) 只对 fp16/bf16 输入支持 fp32 输出，且仅在
            # [CN] CUDA / ROCm 上有实现（后者走非 Lt 的 GEMM）。好处是省掉两份转换：
            # [CN] 既不用把权重 cast 成 fp32 常驻，也不用每步物化 fp32 权重副本。

            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    # [CN] 完整路径：投影 -> （仅 TP>1 时）汇聚 -> 裁掉 padding 词表。
    # [CN] 返回 None 是合法结果，代表本卡不是 gather 的目标 Rank。

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if skip_gather:
            return logits

        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    # [CN] 省通信路径 1：只要全局 argmax。每卡先算本分片最大值，只把 (value, index) 对拿去汇聚，
    # [CN] 通信量 O(batch*2*tp) 而非 O(batch*vocab)。vocab 通常在十万量级，差别是几个数量级。

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        # [CN] 取 argmax 前必须先把分片内的 padding 位（> org_vocab_size 的部分）置成 -inf，
        # [CN] 否则会选到一个词表里根本不存在的 token id。

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        # [CN] 索引必须先转成 float32：bf16 的尾数位不足以精确表示十万量级的 vocab index，
        # [CN] 直接用 bf16 装 index 会得到被舍入过的错误 token id。

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        # [CN] 沿最后一维 all_gather：[b, 2] -> [b, 2*tp]，再 view 成 [b, tp, 2] 后在第 1 维上 argmax，
        # [CN] 得到「哪个 rank 的值最大」，再 gather 出对应的全局 index。

        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    # [CN] 省通信路径 2：只要 top-k。思路同上，但每卡先取本地 k 个候选再汇聚，通信 O(batch*2k*tp)。
    # [CN] 注意 scale / soft_cap 被推迟到最终选出的 k 个值上才作用 —— 二者都是单调变换，
    # [CN] 不改变「哪 k 个最大」的结果，因此延后计算等价且更省。

    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits.

        The `get_top_tokens` reduction widened from one token to k, returning
        the values as well as the global ids. Communication is
        O(batch * 2k * tp_size) rather than O(batch * vocab_size).

        Scale and soft cap are applied to the k selected values rather than
        the whole vocabulary; both are monotonic, so the selection is the same
        and only k entries are touched.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        # [CN] 本地 top-k 之后，把分片内 index 平移到全局词表坐标，再参与跨卡二次 top-k。

        values, ids = _topk(logits, k)
        # Convert shard-local indices to global vocab indices.
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)

        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    # [CN] print(model) 时的摘要行，便于一眼看出 head 的词表与缩放配置。

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
