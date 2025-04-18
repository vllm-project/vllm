# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import logging
import uuid
from ctypes import c_char_p, c_size_t, c_uint32, c_void_p, c_int64
from typing import Optional

from vllm.core.block.prefix_caching_block import PrefixCachingBlock, PrefixHash

logger = logging.getLogger(__name__)


class DynamoResult:
    OK = 0
    ERR = 1


class KVCacheEventManager:

    def __init__(self, namespace: str, component: str, worker_id: int,
                 lib_path: str, kv_block_size: int):
        self.lib = None

        try:
            self.lib = ctypes.CDLL(lib_path)
            self.lib.dynamo_llm_init.argtypes = [
                c_char_p,
                c_char_p,
                c_int64,
                c_uint32,
            ]
            self.lib.dynamo_llm_init.restype = c_uint32

            result = self.lib.dynamo_llm_init(
                namespace.encode(), component.encode(), worker_id, kv_block_size
            )
            if result == DynamoResult.OK:
                logger.info(
                    "KVCacheEventManager initialized successfully. Ready to publish KV Cache Events"
                )
            else:
                logger.info("KVCacheEventManager initialization failed!")

        except Exception as e:
            print(f"Failed to load {lib_path}")
            raise e

        self.lib.dynamo_kv_event_publish_stored.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint32),  # token_ids
            ctypes.POINTER(ctypes.c_size_t),  # num_block_tokens
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
            ctypes.POINTER(ctypes.c_uint64),  # parent_hash
            ctypes.c_uint64,  # lora_id
        ]
        self.lib.dynamo_kv_event_publish_stored.restype = ctypes.c_uint32  # dynamo_llm_result_t

        self.lib.dynamo_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
        ]
        self.lib.dynamo_kv_event_publish_removed.restype = ctypes.c_uint32  # dynamo_llm_result_t

        self.event_id_counter = 0

    def enqueue_stored_event(self, parent: Optional[PrefixCachingBlock],
                             block: PrefixCachingBlock):
        token_ids_arr = (ctypes.c_uint32 *
                         len(block.token_ids))(*block.token_ids)
        num_block_tokens = (ctypes.c_size_t * 1)(len(block.token_ids))
        block_hash = (ctypes.c_uint64 * 1)(block.content_hash)
        parent_hash = ((ctypes.c_uint64 * 1)(parent.content_hash)
                       if parent is not None else None)

        # Publish the event
        result = self.lib.dynamo_kv_event_publish_stored(
            self.event_id_counter,  # uint64_t event_id
            token_ids_arr,  # const uint32_t *token_ids
            num_block_tokens,  # const uintptr_t *num_block_tokens
            block_hash,  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
            parent_hash,  # const uint64_t *parent_hash
            0,  # uint64_t lora_id
        )

        if result == DynamoResult.OK:
            logger.debug(f"Store - Published KV Event: {block.content_hash}")
        else:
            logger.debug(
                f"Store - Failed to Publish KV Event: {block.content_hash}")

        self.event_id_counter += 1

    def enqueue_removed_event(self, block_hash: PrefixHash):
        result = self.lib.dynamo_kv_event_publish_removed(
            self.event_id_counter,
            (ctypes.c_uint64 * 1)(block_hash),
            1,
        )

        if result == DynamoResult.OK:
            logger.debug(f"Remove - Published KV Event: {block_hash}")
        else:
            logger.debug(f"Remove - Failed to Publish KV Event: {block_hash}")

        self.event_id_counter += 1
