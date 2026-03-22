# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存工具模块。

本模块提供了 vLLM V1 引擎的 KV 缓存核心工具函数和数据结构，包括：
- 块哈希（BlockHash）相关类型和操作
- KVCacheBlock：KV 缓存块元数据
- FreeKVCacheBlockQueue：空闲块双向链表队列
- 块哈希计算和验证
- KV 缓存配置生成
- 前缀缓存相关工具

主要类型：
- BlockHash: 单个 KV 缓存块的哈希（用于前缀缓存）
- BlockHashWithGroupId: 带组 ID 的块哈希
- ExternalBlockHash: 外部块哈希类型（用于可重现的块哈希）

主要类：
- KVCacheBlock: KV 缓存块元数据类
- FreeKVCacheBlockQueue: 空闲块队列管理器
- BlockHashListWithBlockSize: 块哈希列表转换器

主要函数：
- hash_block_tokens: 计算块级 token 的哈希
- get_block_hash: 从 BlockHashWithGroupId 提取 BlockHash
- make_block_hash_with_group_id: 打包块哈希和组 ID
- get_kv_cache_configs: 为模型生成 KV 缓存配置
"""

import copy
import hashlib
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, NewType, TypeAlias, overload

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import sha256_cbor, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.utils import tensor_data

# BlockHash 表示用于前缀缓存的单个 KV 缓存块的哈希。
# 将其视为与 bytes 不同的类型有助于避免在传递原始字节字符串时意外误用。
BlockHash = NewType("BlockHash", bytes)

# BlockHashWithGroupId 将 BlockHash 与其 KV 缓存组 ID 组合在一起。
# 它以原始字节表示以保持紧凑和高效。
# 下面的辅助函数将 BlockHash 和组 ID 打包/解包到/从这个键中。
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash 用于可重现的前缀缓存块哈希。
# 它是 bytes 和 int 的联合类型，在我们默认使用 sha256 bytes 进行块哈希后保持向后兼容。
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """将 BlockHash 和组 ID 打包成 BlockHashWithGroupId。

    组 ID 使用 4 字节大端序编码并附加到块哈希字节后面。
    这种表示方式避免了创建元组，同时仍允许我们在需要时恢复两个组件。

    Args:
        block_hash: 块哈希
        group_id: KV 缓存组 ID

    Returns:
        包含组 ID 的块哈希
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """从 BlockHashWithGroupId 中提取 BlockHash。

    Args:
        key: 带组 ID 的块哈希

    Returns:
        纯块哈希
    """
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """从 BlockHashWithGroupId 中提取组 ID。

    Args:
        key: 带组 ID 的块哈希

    Returns:
        组 ID
    """
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    """可能转换块哈希为 ExternalBlockHash 类型。

    根据环境变量决定是否将字节哈希转换为整数哈希。

    Args:
        hash_bytes: 字节类型的块哈希

    Returns:
        ExternalBlockHash 类型（bytes 或 int）
    """
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# 任何前缀块序列的第一个块的哈希种子。
#
# 我们使用随机值来避免哈希冲突，或者如果设置了 PYTHONHASHSEED 环境变量则使用该值，
# 以便进程可以在需要时共享种子。这与 Python 的 hash() 函数的行为一致，
# 后者在未设置 PYTHONHASHSEED 时也使用随机种子。
#
# `init_none_hash` 函数全局初始化这个变量。
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS = frozenset({sha256_cbor, xxhash_cbor})


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    """初始化 NONE_HASH 全局变量。

    Args:
        hash_fn: 哈希函数
    """
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


@dataclass(slots=True)
class KVCacheBlock:
    """KV 缓存块元数据。

    描述 KV 缓存块的属性，包括块 ID、引用计数、块哈希等。
    用于前缀缓存和块管理。

    Attributes:
        block_id: 块 ID，范围从 0 到 num_gpu_blocks - 1
        ref_cnt: 引用计数
        _block_hash: 块的哈希键（块哈希 + 组 ID），仅在块已满并缓存时可用
        prev_free_block: 前一个空闲块（用于双向链表）
        next_free_block: 下一个空闲块（用于双向链表）
        is_null: 是否为空块（不应被缓存）
    """

    # 块 ID，范围从 0 到 num_gpu_blocks - 1
    block_id: int
    # 引用计数
    ref_cnt: int = 0
    # 块的哈希键（块哈希 + 组 ID），仅在块已满并缓存时可用
    _block_hash: BlockHashWithGroupId | None = None

    # 用于构建空闲块的双向链表
    # 这两个属性应仅由 FreeKVCacheBlockQueue 操作
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # 是否为空块（不应被缓存）
    is_null: bool = False

    @property
    def block_hash(self) -> BlockHashWithGroupId | None:
        """获取块哈希。

        Returns:
            块哈希，如果未设置则返回 None
        """
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashWithGroupId):
        """设置块哈希。

        Args:
            block_hash: 要设置的块哈希

        Raises:
            AssertionError: 如果块已经有哈希
        """
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen."
        )
        self._block_hash = block_hash

    def reset_hash(self):
        """重置块哈希（当块被驱逐时调用）。"""
        self._block_hash = None

    def __repr__(self) -> str:
        # 使用 block_id 而不是 KVCacheBlock 对象以避免递归调用 __repr__
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"_block_hash={self._block_hash!r}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


class FreeKVCacheBlockQueue:
    """空闲 KV 缓存块队列。

    将 KVCacheBlock 对象组织成双向链表以管理空闲块。
    实现此类而不是使用 Python 内置的 deque 是为了支持在 O(1) 时间内
    移除队列中间的块。为了缩小与内置 deque（用 C++ 实现）的性能差距，
    此类在操作链表时不分配任何 Python 对象。相反，此类操作给定块的
    prev_free_block 和 next_free_block 属性。

    队列最初按块 ID 排序。当一个块被分配然后释放时，它将按驱逐顺序追加回去：
    1. 最少最近使用的块在前面（LRU）。
    2. 如果两个块具有相同的最后访问时间（由同一序列分配），
       则具有更多哈希 token 的块（块链的尾部）在前面。
    注意：我们通过反转请求的块顺序来维护此顺序。此操作在此类之外。

    Attributes:
        num_free_blocks: 空闲块数量
        fake_free_list_head: 虚拟头节点
        fake_free_list_tail: 虚拟尾节点

    Args:
        blocks: KVCacheBlock 对象列表
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        """初始化空闲块队列。

        Args:
            blocks: KVCacheBlock 对象列表
        """
        self.num_free_blocks = len(blocks)

        # 初始化连续块的双向链接
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # 创建虚拟头节点和尾节点以减少代码分支
        #
        # 实现保证虚拟头尾永远不会被弹出，所以我们可以安全地假设
        # 队列中的每个真实块都有 prev 和 next 块。
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # 将虚拟头和尾分别连接到第一个和最后一个块
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # 对于空列表，直接连接虚拟头和尾
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """弹出第一个空闲块并将 num_free_blocks 减 1。

        Returns:
            第一个空闲块

        Raises:
            ValueError: 如果没有可用空闲块
            RuntimeError: 如果块无效
        """
        if (
            self.fake_free_list_head.next_free_block is self.fake_free_list_tail
            or self.fake_free_list_head.next_free_block is None
        ):
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        first_block: KVCacheBlock = self.fake_free_list_head.next_free_block

        if first_block.next_free_block is None:
            # 如果块来自空闲列表，这不应该发生。
            # 它表示调用者逻辑中的 bug。
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # 连接虚拟头和第一个块的下一个块（即第二个块或虚拟尾）
        self.fake_free_list_head.next_free_block = first_block.next_free_block
        first_block.next_free_block.prev_free_block = self.fake_free_list_head

        # 从链表中移除块
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """弹出前 n 个空闲块并将 num_free_blocks 减 n。

        Args:
            n: 要弹出的块数量

        Returns:
            n 个空闲块列表
        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        # 从列表头部弹出 n 个块
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # 重置所有弹出块的 prev_free_block 和 next_free_block
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # 队列不为空，将虚拟头连接到新第一个块
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """从空闲列表中移除一个块并将 num_free_blocks 减 1。

        Args:
            block: 要移除的块

        Raises:
            RuntimeError: 如果块无效
        """
        if block.prev_free_block is None or block.next_free_block is None:
            # 如果块来自空闲列表，这不应该发生。
            # 它表示调用者逻辑中的 bug。
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # 将前一个块连接到后一个块
        block.prev_free_block.next_free_block = block.next_free_block
        # 将后一个块连接到前一个块
        block.next_free_block.prev_free_block = block.prev_free_block

        # 从链表中移除块
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """将块放回空闲列表并将 num_free_blocks 加 1。

        Args:
            block: 要追加的块

        Raises:
            RuntimeError: 如果虚拟尾节点的 prev_free_block 不存在
        """
        if self.fake_free_list_tail.prev_free_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        last_block: KVCacheBlock = self.fake_free_list_tail.prev_free_block

        # 在最后一个块后面连接新块
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # 在新块后面连接虚拟尾
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block

        self.num_free_blocks += 1

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """将一组块放回空闲列表。

        Args:
            blocks: 要追加的块列表
        """
        if len(blocks) == 0:
            return

        last_block = self.fake_free_list_tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # 在 <blocks> 的块之间添加互连
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # 将 <blocks> 的最后一个块连接到虚拟尾
        last_block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """获取空闲列表中的所有空闲块。主要用于测试。

        Returns:
            空闲块列表
        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # 从第一个块开始
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # 只要 next_free_block 可用，就还没到达虚拟尾
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """检查请求分配的块是否需要额外的哈希键。

    多模态请求需要包含 MM 哈希。
    LoRA 请求需要包含 LoRA 名称。
    提供了 cache salt 的请求需要包含 salt。

    Args:
        request: 请求

    Returns:
        是否需要额外哈希键
    """
    return (
        bool(request.mm_features)
        or (request.lora_request is not None)
        or (request.cache_salt is not None)
    )


def _gen_mm_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[list[Any], int]:
    """为多模态请求生成块哈希计算的额外键。

    对于多模态输入，额外键是 (mm_hash, start_offset)，
    表示块中包含的多模态输入及其在块 token 中的起始偏移量。

    Args:
        request: 请求
        start_token_idx: 块的起始 token 索引
        end_token_idx: 块的结束 token 索引
        start_mm_idx: 块的起始多模态索引

    Returns:
        额外键元组和下一个多模态索引
    """
    extra_keys: list[Any] = []

    mm_features = request.mm_features
    if not mm_features:
        return extra_keys, start_mm_idx

    # 注意：假设 mm_features 按 mm_position.offset 排序。
    # 如果起始 token 索引超出范围，我们不需要检查所有 mm 输入。
    # 这通常发生在 prefill 后期和解码阶段。
    last_pos = mm_features[-1].mm_position
    if last_pos.offset + last_pos.length <= start_token_idx:
        return extra_keys, start_mm_idx

    # 支持 start_mm_idx == -1 表示最后一个 mm 输入
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_features)
        start_mm_idx = len(mm_features) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_features and curr_mm_idx < len(mm_features):
        mm_feature = mm_features[curr_mm_idx]
        assert mm_feature.identifier is not None
        offset = mm_feature.mm_position.offset
        length = mm_feature.mm_position.length
        if end_token_idx > offset:
            if start_token_idx >= offset + length:
                # 此块已 passed 当前 mm 输入
                curr_mm_idx += 1
                continue

            # 块包含当前 mm 输入。包含其相对于块起始的偏移量，
            # 以便当相同 MM 项目出现在其他相同占位符块中的不同位置时，
            # 前缀缓存键保持不同。
            extra_keys.append((mm_feature.identifier, offset - start_token_idx))

            if end_token_idx >= offset + length:
                # 如果此块包含当前 mm 输入的结尾，移到下一个 mm 输入
                curr_mm_idx += 1
            else:
                # 否则此块已完成 mm 输入
                break
        else:
            # 此块还未到达当前 mm 输入
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[str]:
    """为 LoRA 请求生成块哈希计算的额外键。

    Args:
        request: 请求

    Returns:
        如果是 LoRA 请求则返回 LoRA 名称，否则返回空列表
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]


def _gen_prompt_embeds_extra_hash_keys(
    request: Request, start_token_idx: int, end_token_idx: int
) -> list[bytes]:
    """为 prompt embeddings 生成块哈希计算的额外键。

    Args:
        request: 请求
        start_token_idx: 块的起始 token 索引
        end_token_idx: 块的结束 token 索引

    Returns:
        如果存在 prompt embeddings 则返回块的稳定哈希，否则返回空列表
    """
    if request.prompt_embeds is None:
        return []
    block_range = (start_token_idx, end_token_idx)
    embeds_hash = request._prompt_embeds_per_block_hashes.get(block_range)
    if embeds_hash is None:
        block_prompt_embeds = request.prompt_embeds[start_token_idx:end_token_idx]
        # 每个块哈希一次 prompt embeddings 并缓存到请求上
        embeds_hash = hashlib.sha256(tensor_data(block_prompt_embeds)).digest()
        request._prompt_embeds_per_block_hashes[block_range] = embeds_hash
    return [embeds_hash]


def generate_block_hash_extra_keys(
    request: Request, start_token_idx: int, end_token_idx: int, start_mm_idx: int
) -> tuple[tuple[Any, ...] | None, int]:
    """生成块哈希的额外键。

    额外键可以来自多模态输入、请求特定元数据（如 LoRA 名称）
    和 prompt embeddings 的哈希数据。

    Args:
        request: 请求
        start_token_idx: 块的起始 token 索引
        end_token_idx: 块的结束 token 索引
        start_mm_idx: 块的起始多模态索引

    Returns:
        额外键元组和下一个多模态索引
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx
    )
    lora_extra_keys: list[str] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = (
        [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []
    )
    prompt_embeds_keys = _gen_prompt_embeds_extra_hash_keys(
        request, start_token_idx, end_token_idx
    )

    extra_keys: list[Any] = (
        lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys
    )

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """计算块内容的哈希值。

    计算对应于块内容和前导块内容的哈希值。
    哈希值用于前缀缓存。我们使用 LRU 缓存来避免为相同块内容重复计算哈希值。

    Args:
        hash_function: 用于计算块哈希的哈希函数
        parent_block_hash: 父块的哈希。如果是第一个块则为 None
        curr_block_token_ids: 当前块中的 token id 列表。当前块假定已满
        extra_keys: 块的额外键

    Returns:
        块的哈希值
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )


def get_request_block_hasher(
    block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
) -> Callable[[Request], list[BlockHash]]:
    """返回一个函数，用于计算请求的未计算块哈希列表。

    Args:
        block_size: 块大小
        caching_hash_fn: 缓存哈希函数

    Returns:
        计算请求块哈希的函数
    """

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * block_size
        num_tokens = request.num_tokens

        if start_token_idx + block_size > num_tokens:
            # 当没有新完整块创建时提前停止
            return []

        curr_mm_idx = 0
        if start_token_idx > 0:
            # 设置 curr_mm_idx = -1 表示最后一个 mm 输入
            # 注意：因为我们只在这分支中当块由生成的 token 完成时，
            # 我们只需要考虑最后一个 mm 输入
            curr_mm_idx = -1

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + block_size
            if end_token_idx > num_tokens:
                # 我们只哈希完整块
                break

            # MM 和 LoRA 请求需要额外键进行块哈希计算
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start_token_idx, end_token_idx, curr_mm_idx
            )

            # 计算当前块的哈希
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    """检查是否有足够的 KV 缓存内存。

    Args:
        available_memory: 可用内存
        get_needed_memory: 获取所需内存的函数
        max_model_len: 最大模型长度
        estimate_max_model_len: 估计最大模型长度的函数

    Raises:
        ValueError: 如果没有足够的内存
    """
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine. "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, kv_cache_specs: Iterable[KVCacheSpec]
) -> int:
    """获取给定 KV 缓存规范的最大内存使用量（字节）。

    Args:
        vllm_config: vLLM 配置
        kv_cache_specs: KV 缓存规范迭代器

    Returns:
        最大内存使用量（字节）
    """
    return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
) -> int:
    """使用二分搜索估计可适应可用内存的最大模型长度。

    此函数在估计期间临时修改 max_model_len，但在返回前恢复原始值，
    确保无副作用。

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_spec: 模型中每个注意力层的 kv cache spec
        available_memory: 可用于 KV 缓存的内存（字节）

    Returns:
        可适应可用内存的估计最大模型长度
    """
    # 保存原始 max_model_len 以便在估计后恢复
    original_max_model_len = vllm_config.model_config.max_model_len

    # 定义一个函数来检查给定模型长度是否适合内存
    def fits_in_memory(model_len: int) -> bool:
        # 临时修改 max_model_len 用于此计算
        vllm_config.model_config.max_model_len = model_len
        # 计算给定模型长度所需的内存
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # 二分搜索最大模型长度
        left, right = 1, original_max_model_len

        # 如果即使最小的模型长度也不适合，返回 0
        if not fits_in_memory(left):
            return 0

        # 二分搜索适合的最大模型长度
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # 始终恢复原始 max_model_len 以避免副作用
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    available_memory: int,
):
    """检查 available_memory 是否足够 KV 缓存至少一个请求。

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_spec: 模型中每个注意力层的 kv cache spec
        available_memory: 可用于 KV 缓存的内存（字节）

    Raises:
        ValueError: 如果没有足够的内存用于 KV 缓存
    """
    # 如果 kv_cache_spec 为空则不需要检查可用内存
    if kv_cache_spec:
        _check_enough_kv_cache_memory(
            available_memory,
            lambda: max_memory_usage_bytes(vllm_config, kv_cache_spec.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, kv_cache_spec, am),
        )


def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """创建 KVCacheGroupSpec 对象。

    同一组中的层应共享相同的 KVCacheSpec。

    Args:
        kv_cache_spec: 从每个层名称到其对应 KVCacheSpec 的映射
        grouped_layer_names: KV 缓存组列表，每个元素是一个层名称列表，
            这些层属于同一组并应共享相同的 KVCacheSpec

    Returns:
        KVCacheGroupSpec 对象列表，每组一个
    """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups


def is_kv_cache_spec_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """检查 KVCacheSpec 中的所有层是否具有相同的类型。

    注意：我们将带有和不带有 sliding window 的 FullAttentionSpec 视为同一类型。

    Args:
        kv_cache_spec: 模型中每个注意力层的 kv cache spec

    Returns:
        如果所有层类型相同则返回 True，否则返回 False
    """
    if not kv_cache_spec:
        # 仅编码器模型没有 KV 缓存，kv_cache_type 可视为统一
        return True
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """获取给定 KV 缓存配置的最大并发数。

    Args:
        vllm_config: vLLM 配置
        kv_cache_config: KV 缓存配置

    Returns:
        最大并发数
    """
    num_layer_per_group = max(
        len(group.layer_names) for group in kv_cache_config.kv_cache_groups
    )
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(
        vllm_config, (group.kv_cache_spec for group in kv_cache_config.kv_cache_groups)
    )
    memory_per_block = (
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        * num_layer_per_group
    )
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """如果设置了 num_gpu_blocks_override 则覆盖块数量。

    Args:
        vllm_config: vLLM 配置
        num_blocks: 原始块数量

    Returns:
        可能被覆盖的块数量
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            num_blocks,
            num_gpu_blocks_override,
        )
        num_blocks = num_gpu_blocks_override

    return num_blocks


def get_num_blocks(
    vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int
) -> int:
    """获取 KV 缓存块数量。

    Args:
        vllm_config: 全局 VllmConfig
        num_layers: 层数量
        available_memory: 可用于 KV 缓存的内存（字节）
        page_size: KV 缓存的页大小

    Returns:
        KV 缓存块数量
    """
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """获取 KV 缓存的页大小。

    Args:
        kv_cache_specs: KV 缓存规范迭代器

    Returns:
        统一的页大小
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def _get_kv_cache_groups_uniform_spec(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """为所有层具有相同 KV 缓存 spec 的模型生成 KV 缓存配置。

    Args:
        kv_cache_specs: 模型中每个注意力层的 kv cache spec

    Returns:
        生成的 KVCacheGroupSpecs
    """
    return create_kv_cache_group_specs(kv_cache_specs, [list(kv_cache_specs.keys())])


def _get_kv_cache_groups_uniform_type(
    spec: UniformTypeKVCacheSpecs,
) -> list[KVCacheGroupSpec]:
    """为所有层具有相同 KV 缓存类型但不同 hidden sizes 的模型生成配置。

    所有层合并为一组。

    Args:
        spec: 模型的 UniformTypeKVCacheSpecs

    Returns:
        生成的 KVCacheGroupSpecs
    """
    return [KVCacheGroupSpec(list(spec.kv_cache_specs.keys()), spec)]


def is_kv_cache_page_size_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """检查 KVCacheSpec 中的所有层是否具有相同的页大小。

    Args:
        kv_cache_spec: 模型中每个注意力层的 KVCacheSpec

    Returns:
        如果所有层页大小相同则返回 True，否则返回 False
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    return len(page_sizes) == 1


def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """统一给定 KVCacheSpec 的页大小。

    如果所有层的页大小相同，返回原始 KVCacheSpec。
    如果不同，通过增加具有较小页大小的层的块大小来统一页大小。
    如果无法统一，则抛出 NotImplementedError。

    Args:
        kv_cache_spec: 模型中每个注意力层的 KVCacheSpec

    Returns:
        具有相同 page_size_bytes 的更新后的 KVCacheSpec

    Raises:
        NotImplementedError: 如果无法统一页大小
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # 所有层具有相同的页大小，无需统一
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec


def is_kv_cache_type_attention_free(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """检查是否为无注意力模型（kv_cache_spec 为空字典）。

    Args:
        kv_cache_spec: KVCacheSpec 字典

    Returns:
        如果为无注意力模型则返回 True
    """
    return not kv_cache_spec


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """为具有多种注意力类型但统一页大小的混合模型生成 KV 缓存组。

    混合模型 KV 缓存管理的详细说明：
    模型中的层以某种模式重复，例如，具有 10 个 full attention 层和
    20 个 sliding window attention 层的模型可以看作是重复模式 (1 * full, 2 * sw) 10 次。
    KVCacheManager 为模式中的每个层分配不同的块表，并重复它们 10 次
    以生成模型中 30 层的块表。
    因此，我们可以将模型中的层分为 3 个 kv_cache_groups，每组包含模型中的 10 层。
    KVCacheManager 为每组基于其 kv_cache spec 分配块表，
    模型运行器将块表应用于组中的每一层。

    示例：
    1. 仅使用 full attention 的模型。模式为 (num_hidden_layers * full)，
       所以只有一组，所有层共享块表。
    2. 具有 10 个 full attention 层和 20 个 sliding window attention 层的模型。
       模式中有 3 层 (1 * full, 2 * sw)，所以有 3 个 kv_cache_groups，每组代表 10 层。

    为简化实现，我们做出以下假设：
    1. 每块物理内存：所有 KV 缓存组必须相同。违反此假设是非平凡的，
       因为分配不同大小的块时存在内存碎片问题。
    2. 每块 token 数（block_size）：目前我们直接对所有层使用 CacheConfig.block_size。
       可以扩展为按 KV 缓存组变化，但在每个 KV 缓存组内，所有层必须共享相同的块大小。
    3. 每层每 token 物理内存：此属性由模型配置决定。目前我们只支持
       所有层具有相同每层每 token 物理内存的模型。
    4. 每组层数：目前假设所有层相同。
    5. 组内注意力类型：组内所有层必须共享相同的注意力类型。
    6. 支持多种注意力类型：find_longest_cache_hit 仅支持一种注意力类型
       或两种类型（full-attention 加上另一个类型）。

    Args:
        kv_cache_spec: 模型中每个注意力层的 KVCacheSpec

    Returns:
        生成的 KVCacheGroupSpecs
    """
    # 按 kv_cache_spec 对所有层分组
    same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)

    # 将每个组分割成更小的组，使每组的层数相同
    # 必要时在每组的最后一组添加填充层
    min_num_layers = min([len(layers) for layers in same_type_layers.values()])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in same_type_layers.values()])
    if max_num_layers < min_num_layers * 1.5:
        # 如果层数不比最小层数大很多，使用最大层数作为组大小
        # 以避免太多填充层
        group_size = max_num_layers
    grouped_layers = []
    for layers in same_type_layers.values():
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # 在 PP 情况下，如果我们有：
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # 我们应该有 3 组：(full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # 不能是 (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3)
        # 因为 stage 0 的 3 组将是 (full.0), (sw.0, sw.1), (empty group)
        # 并会被填充到 (full.0, padding), (sw.0, sw.1), (padding, padding)
        # 以确保每组层数相同，但这会导致内存浪费
        # 为避免这种情况，我们分配 layers[i::num_groups] 给第 i 组
        # 而不是 layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """从 KV 缓存组生成 KV 缓存配置。

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_groups: KV 缓存组
        available_memory: 可用于 KV 缓存的内存（字节）

    Returns:
        生成的 KVCacheConfig
    """
    if len(kv_cache_groups) == 0:
        # 无注意力模型没有 KV 缓存
        # 返回 num_blocks=1 因为 BlockPool 总是需要一个 null_block
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )

    # 确定模型运行器应如何初始化 KV 缓存张量
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        # 特殊情况：所有层具有相同类型的 KV 缓存但 hidden size 不同
        # 基于每层的 hidden size 分配不同的内存量
        num_blocks = (
            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes
        )
        num_blocks = may_override_num_blocks(vllm_config, num_blocks)
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        kv_cache_tensors = [
            KVCacheTensor(
                size=per_layer_specs[layer_name].page_size_bytes * num_blocks,
                shared_by=[layer_name],
            )
            for layer_name in kv_cache_groups[0].layer_names
        ]
    else:
        # 一般情况：
        # 我们将有 group_size 个内存池，每个池由每组中的一个层共享
        # 由于不同组的层有不同的块表，它们将使用共享 Tensor 的不同部分
        # 3 组 (full.0, full.1), (sw.0, sw.2), (sw.1, padding) 的内存布局为：
        # full.0, sw.0, sw.1: 共享大小为 available_memory//2 的 Tensor
        # full.1, sw.2: 共享另一个大小为 available_memory//2 的 Tensor
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """统一混合模型的 KV 缓存 spec。

    如果模型是具有多种 KV 缓存类型的混合模型，此函数尝试将所有
    KV 缓存 spec 转换为一种类型。如果同时存在 SlidingWindowSpec 和
    FullAttentionSpec，则将 SlidingWindowSpec 转换为 FullAttentionSpec。

    Args:
        kv_cache_spec: 模型中每个注意力层的 kv cache spec
    """
    if is_kv_cache_spec_uniform(
        kv_cache_spec
    ) or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec):
        return

    logger.warning(
        "Hybrid KV cache manager is disabled for this hybrid model, "
        "This means we do not enable any optimizations for saving KV cache "
        "memory (e.g., dropping the KV cache outside the sliding window). "
        "The compute of layers like sliding window is still saved."
    )

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values()
    )
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values()
    )
    has_chunked_local_attention = any(
        isinstance(spec, ChunkedLocalAttentionSpec) for spec in kv_cache_spec.values()
    )
    if has_full_attention and (has_sliding_window or has_chunked_local_attention):
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                    page_size_padded=spec.page_size_padded,
                )
            elif isinstance(spec, ChunkedLocalAttentionSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    attention_chunk_size=spec.attention_chunk_size,
                    page_size_padded=spec.page_size_padded,
                )

    if not (
        is_kv_cache_spec_uniform(kv_cache_spec)
        or UniformTypeKVCacheSpecs.is_uniform_type(kv_cache_spec)
    ):
        raise ValueError(
            "Hybrid KV cache manager is disabled but failed to "
            "convert the KV cache specs to one unified type."
        )


def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
    """将模型中的层分成具有相同 KV 缓存 spec 的组。

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_spec: 模型中每个注意力层的 kv cache spec

    Returns:
        生成的 KVCacheGroups
    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)

    if is_kv_cache_type_attention_free(kv_cache_spec):
        # 返回空列表以允许 KVCacheManager 处理无注意力模型
        return []

    if is_kv_cache_spec_uniform(kv_cache_spec):
        # 所有层的 KV 缓存相同，这适用于大多数模型
        # 为每层分配相同的内存
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # 所有层需要相同数量的 token 槽位
        # 将所有层放入一组
        return _get_kv_cache_groups_uniform_type(uniform_spec)

    # 由于 KVCacheManager 只能分配一种大小的内存，我们需要统一层的页大小
    # 对于无法统一的情况，此函数将抛出错误
    kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
    # 模型包含多种注意力类型，但所有层的每块每层物理内存相同
    # 将层分成具有相同层数的组，从而具有相同的总页大小
    return _get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """为调度器生成 KV 缓存配置。

    Args:
        kv_cache_configs: KV 缓存配置列表

    Returns:
        调度器的 KVCacheConfig
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # 所有工作节点具有相同的 kv_cache_config（除了层名称），
    # 所以使用任意一个来初始化调度器
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # UniformTypeKVCacheSpecs 中的所有层类型相同，
            # 所以使用任意一个来初始化调度器
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def _report_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """记录已解析的 KV 缓存配置。

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_config: 已解析的 KV 缓存配置
    """
    min_block_size = min(
        [group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups]
    )

    # 记录 KV 缓存大小和最大并发数
    num_tokens = (
        kv_cache_config.num_blocks
        // len(kv_cache_config.kv_cache_groups)
        * min_block_size
    )
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size * dcp_size > 1:
        num_tokens *= pcp_size * dcp_size
        logger.info(
            "Multiplying the GPU KV cache size by the cp_world_size %d "
            "(pcp_world_size %d * dcp_world_size %d).",
            pcp_size * dcp_size,
            pcp_size,
            dcp_size,
        )
    num_tokens_str = f"{num_tokens:,}"
    logger.info_once("GPU KV cache size: %s tokens", num_tokens_str, scope="local")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    logger.info_once(
        "Maximum concurrency for %s tokens per request: %.2fx",
        max_model_len_str,
        max_concurrency,
        scope="local",
    )


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """从 KV 缓存组计算最大内存使用量（字节）。

    正确处理混合模型中的填充。例如，如果模型有 8 个 full attention 层
    和 9 个 sliding window 层，它们将被填充为 9 个 full + 9 个 sliding window
    以统一组大小。

    Args:
        vllm_config: vLLM 配置
        kv_cache_groups: KV 缓存组

    Returns:
        最大内存使用量（字节）
    """
    if not kv_cache_groups:
        return 0

    # UniformTypeKVCacheSpecs 特殊情况（单组，每层 spec）
    if len(kv_cache_groups) == 1 and isinstance(
        kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
    ):
        per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
        return sum(
            spec.max_memory_usage_bytes(vllm_config)
            for spec in per_layer_specs.values()
        )

    # 一般情况：group_size 个池，每个池由每组的每层共享
    # 内存 = group_size * page_size * blocks_for_max_len
    group_size = max(len(group.layer_names) for group in kv_cache_groups)
    page_size = get_uniform_page_size(
        [group.kv_cache_spec for group in kv_cache_groups]
    )
    blocks_needed = sum(
        cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config), page_size)
        for group in kv_cache_groups
    )

    return group_size * page_size * blocks_needed


def _estimate_max_model_len_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """二分搜索适合可用内存的最大模型长度。

    如果即使 1 个 token 也不适合则返回 0。

    Args:
        vllm_config: vLLM 配置
        kv_cache_groups: KV 缓存组
        available_memory: 可用内存

    Returns:
        最大模型长度
    """
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        return (
            _max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """当 max_model_len 设置为 -1 时，自动估计可支持的最大上下文长度。

    使用二分搜索找到适合所有工作节点的最大长度。

    Args:
        vllm_config: 全局 VllmConfig（将原地修改）
        projected_groups_per_worker: 每个工作节点的 KV 缓存组投影
        available_memory: 每个工作节点的可用内存（字节）

    Raises:
        ValueError: 如果没有足够的 GPU 内存
    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # 所有工作节点都没有组（无注意力模型）
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
            scope="local",
        )
        return

    # 找到适合所有工作节点的 max_model_len
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(vllm_config, groups, avail_mem)
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    if auto_fit_max >= original_max:
        # 模型的完整上下文长度适合内存
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available GPU memory",
            original_max,
            scope="local",
        )
    else:
        # 需要减少 max_model_len 以适应内存
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available GPU memory (%s GiB available for KV cache)",
            original_max,
            auto_fit_max,
            format_gib(limiting_worker_mem),
            scope="local",
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """将全局 KV 缓存组投影到单个工作节点的已分配层。

    在流水线并行中，每个工作节点只拥有层的一个子集。此函数过滤
    全局组以仅包含给定工作节点上存在的层，并相应地调整
    UniformTypeKVCacheSpecs。

    Args:
        global_kv_cache_groups: 整个模型的全局 KV 缓存组
        worker_spec: 此工作节点上每层的 KV cache spec

    Returns:
        仅包含此工作节点层的投影 KV 缓存组
    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(KVCacheGroupSpec(worker_layer_names, group_spec))
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """为模型生成 KV 缓存配置。

    由于我们使用共享的集中式控制器管理所有工作节点，我们需要 kv_cache_config
    在所有工作节点之间保持一致，以确保 KV 缓存分配可以应用于所有工作节点。
    然而，不同的工作节点可能有不同的可用内存和不同的层类型（启用流水线并行时）。
    为了处理工作节点之间的差异，当前实现是：
    1. 合并所有工作节点的 KV cache specs 以获得整个模型的 KVCacheSpecs
    2. 基于整个模型的层比率生成 KV 缓存组
    3. 使用每工作节点投影组处理自动拟合 max_model_len 和内存检查
    4. 基于 KV 缓存分组策略为每个工作节点生成 KV 缓存配置
    5. 将每个工作节点的 num_blocks 更改为所有工作节点中最小的，
       并按比例缩小张量大小以避免分配未使用的内存

    Args:
        vllm_config: 全局 VllmConfig
        kv_cache_specs: 每个工作节点的 KVCacheSpec 列表
        available_memory: 每个工作节点的可用内存（字节）

    Returns:
        每个工作节点的 KVCacheConfigs
    """
    # 合并所有工作节点的 KV cache specs
    merged_kv_cache_specs: dict[str, KVCacheSpec] = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            if layer_name not in merged_kv_cache_specs:
                merged_kv_cache_specs[layer_name] = layer_spec
            else:
                assert merged_kv_cache_specs[layer_name] == layer_spec, (
                    "The KV cache specs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # 获取全局 KV 缓存组
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # 如果 original_max_model_len 为 -1，自动确定适合可用 GPU 内存的最大模型长度
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in kv_cache_specs
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            vllm_config, projected_groups_per_worker, available_memory
        )

    # 检查每个工作节点的内存是否足够
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, kv_cache_specs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            kv_cache_spec_one_worker
        ), "Some layers are not assigned to any group."
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # 将每个 rank 的 num_blocks 更改为所有 rank 中最小的
    # 我们还需要按比例缩小张量大小以避免分配未使用的内存
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # 按比例缩小张量大小
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks

        if len(kv_cache_config.kv_cache_groups) > 0:
            _report_kv_cache_config(vllm_config, kv_cache_config)

    return kv_cache_configs


class BlockHashListWithBlockSize:
    """块哈希列表转换器（按块大小）。

    将块哈希粒度从 hash_block_size 转换为 target_block_size。
    当 KV 缓存组具有不同块大小时使用：hash_block_size 是计算原始 block_hashes
    时使用的大小；target_block_size 是组的实际块大小。

    目前仅支持按整数因子缩放（即 target_block_size 是 hash_block_size 的倍数）。
    转换在访问时惰性执行以提高效率，方法是连接 hash_block_size 的连续哈希
    以形成 target_block_size 的每个哈希。

    示例（hash_block_size = 16，target_block_size = 32）：
    连接两个 16 大小的哈希得到一个 32 大小的哈希：

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | AB   | CD    |

    Args:
        block_hashes: 要转换的块哈希，在 hash_block_size 计算
        hash_block_size: 计算 block_hashes 时使用的块大小
        target_block_size: 目标块大小；必须是 hash_block_size 的倍数
    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        """初始化块哈希列表转换器。

        Args:
            block_hashes: 要转换的块哈希列表
            hash_block_size: 源块大小
            target_block_size: 目标块大小
        """
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        """返回转换后的哈希列表长度。

        Returns:
            转换后的哈希数量
        """
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        """获取索引处的块哈希。

        Args:
            idx: 索引或切片

        Returns:
            块哈希或块哈希列表

        Raises:
            TypeError: 如果索引类型无效
        """
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        """迭代转换后的块哈希。

        Yields:
            转换后的块哈希
        """
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        """获取指定索引处的转换后哈希值。

        Args:
            idx: 索引

        Returns:
            合并后的块哈希
        """
        base = idx * self.scale_factor
        end = base + self.scale_factor
        merged_hash: bytes = self.block_hashes[base]
        for i in range(base + 1, end):
            merged_hash += self.block_hashes[i]
        return BlockHash(merged_hash)


# BlockHashList 类型别名
BlockHashList = list[BlockHash] | BlockHashListWithBlockSize
