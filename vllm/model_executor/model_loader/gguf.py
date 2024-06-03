# adapted from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/integrations/ggml.py
# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Integration with GGML / The file is copied and adapted from https://github.com/99991/pygguf
with extra methods beings exposed
"""
import torch
import numpy as np
from transformers.integrations.ggml import GGML_BLOCK_SIZES, GGML_TYPES, load_dequant_gguf_tensor


def convert_tensor_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(
        num_blocks, block_size // 2
    )
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xF).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack
    quants = np.hstack([ql, qr])

    return scales, quants


def convert_tensor_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = len(data) // block_size

    scales = (
        np.frombuffer(data, dtype=np.float16)
        .reshape(num_blocks, 1 + 16)[:, :1]
        .astype(np.float32)
    )
    quants = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales, quants


def load_gguf_tensor(tensor):
    shape, ggml_type, data = tensor.shape, tensor.tensor_type, tensor.data

    scales = None
    if ggml_type == GGML_TYPES["Q8_0"] and "blk" in tensor.name:
        scales, quants = convert_tensor_q8_0(data)
    elif ggml_type == GGML_TYPES["Q4_0"] and "blk" in tensor.name:
        scales, quants = convert_tensor_q4_0(data)
    else:
        quants = load_dequant_gguf_tensor(shape, ggml_type, data)
        quants = torch.from_numpy(quants.copy())
        return scales, quants
    
    scales_shape = (int(shape[0]//32), shape[1])
    scales = scales.reshape(scales_shape[::-1])
    quants = quants.reshape(shape[::-1])
    scales = torch.from_numpy(scales.copy())
    quants = torch.from_numpy(quants.copy())
    return scales, quants
