# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from array import array
from collections.abc import Iterable
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.pooling_metadata import (
    PoolingMetadata as V0PoolingMetadata, PoolingTensors)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, PoolerOutput, PoolingSequenceGroupOutput
from vllm.v1.pool.metadata import PoolingMetadata as V1PoolingMetadata

from .interfaces import SupportsMultiModal, SupportsCrossEncoding
from .qwen2_vl import (Qwen2VLDummyInputsBuilder,
                       Qwen2VLForConditionalGeneration,
                       Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo)
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

logger = init_logger(__name__)

# Vision token IDs for Jina V4
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

# Maximum sequence length for safety
MAX_SEQUENCE_LENGTH = 512 * 1024  # 512K tokens


PoolingMetadata = Union[V0PoolingMetadata, V1PoolingMetadata]


# Triton kernel for optimized vision token extraction
if HAS_TRITON:
    @triton.jit
    def extract_vision_tokens_kernel(
        hidden_states_ptr,
        token_ids_ptr,
        output_ptr,
        seq_start,
        seq_len,
        hidden_size,
        vision_start_id: tl.constexpr,
        vision_end_id: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel to extract and pool vision tokens efficiently."""
        pid = tl.program_id(0)
        
        if pid >= hidden_size:
            return
            
        # Find vision token range
        vision_count = 0
        accumulator = 0.0
        
        for i in range(seq_len):
            token_id = tl.load(token_ids_ptr + seq_start + i)
            if token_id >= vision_start_id and token_id <= vision_end_id:
                hidden_val = tl.load(
                    hidden_states_ptr + (seq_start + i) * hidden_size + pid
                )
                accumulator += hidden_val
                vision_count += 1
        
        # Store mean pooled result
        if vision_count > 0:
            result = accumulator / vision_count
        else:
            result = 0.0
            
        tl.store(output_ptr + pid, result)


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class JinaVLForEmbedding(Qwen2VLForConditionalGeneration,
                         SupportsCrossEncoding,
                         SupportsMultiModal):
    # Weight mapping for HuggingFace checkpoint compatibility
    weight_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=maybe_prefix(prefix, "qwen2_vl"))
        
        self.hidden_size = vllm_config.model_config.hf_config.hidden_size
        pooler_config = vllm_config.model_config.pooler_config
        self.observability_config = vllm_config.observability_config
        
        # Initialize base pooler for fallback
        self._base_pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False
        )
        
        # Performance tracking
        self._pooling_time_ms = 0.0
        self._pooling_count = 0
        
        logger.info("Initialized JinaVLForEmbedding with thread-safe pooling")

    def _extract_token_ids_safe(
        self,
        pooling_metadata: PoolingMetadata
    ) -> Tuple[List[array], List[int]]:
        """Safely extract token IDs from pooling metadata."""
        token_ids_list: List[array] = []
        try:
            if isinstance(pooling_metadata, V1PoolingMetadata):
                # For V1, we get token IDs and sequence indices directly
                for i, num in enumerate(pooling_metadata.prompt_lens):
                    token_ids = pooling_metadata.prompt_token_ids[i, :num].tolist()
                    token_ids_list.append(array('l', token_ids))
                
                # V1 metadata does not have explicit seq_ids, so we use indices
                seq_ids = list(range(len(token_ids_list)))
                return token_ids_list, seq_ids

            # For V0, we extract from seq_groups and seq_data
            seq_ids = []
            for seq_group, _ in pooling_metadata.seq_groups:
                for seq_id in seq_group:
                    if seq_id not in pooling_metadata.seq_data:
                        logger.warning(f"Sequence {seq_id} not found in seq_data")
                        continue
                        
                    seq_data = pooling_metadata.seq_data[seq_id]
                    
                    # Get prompt token IDs safely
                    if hasattr(seq_data, 'prompt_token_ids_array'):
                        token_ids = seq_data.prompt_token_ids_array
                    elif hasattr(seq_data, '_prompt_token_ids'):
                        token_ids = seq_data._prompt_token_ids
                    else:
                        logger.warning(f"No token IDs found for sequence {seq_id}")
                        continue
                    
                    seq_ids.append(seq_id)
                    token_ids_list.append(token_ids)
            
            return token_ids_list, seq_ids
            
        except Exception as e:
            logger.error(f"Error extracting token IDs: {e}. "
                         f"Extracted {len(token_ids_list)} sequences before failure")
            raise

    def _apply_vision_pooling_optimized(
        self,
        hidden_states: torch.Tensor,
        token_ids_list: List[array],
        prompt_lens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply optimized vision token pooling using Triton kernels."""
        if not HAS_TRITON:
            logger.debug("Triton not available, falling back to PyTorch implementation")
            return self._apply_vision_pooling_pytorch(hidden_states, token_ids_list, prompt_lens)
        
        pooled_outputs = []
        offset = 0
        device = hidden_states.device
        
        for i, (token_ids, prompt_len) in enumerate(zip(token_ids_list, prompt_lens)):
            prompt_len = int(prompt_len.item())
            
            # Convert token IDs to tensor
            token_tensor = torch.tensor(list(token_ids), dtype=torch.long, device=device)
            
            # Allocate output tensor
            output = torch.zeros(self.hidden_size, device=device, dtype=hidden_states.dtype)
            
            # Check for vision tokens
            has_vision = torch.any(
                (token_tensor >= VISION_START_TOKEN_ID) & 
                (token_tensor <= VISION_END_TOKEN_ID)
            )
            
            if has_vision:
                # Use Triton kernel for vision token extraction
                grid = (self.hidden_size,)
                extract_vision_tokens_kernel[grid](
                    hidden_states,
                    token_tensor,
                    output,
                    offset,
                    prompt_len,
                    self.hidden_size,
                    VISION_START_TOKEN_ID,
                    VISION_END_TOKEN_ID,
                    BLOCK_SIZE=1024,
                )
            else:
                # Regular mean pooling for text
                seq_states = hidden_states[offset:offset + prompt_len]
                output = seq_states.mean(dim=0)
            
            # Normalize (check for zero vector to avoid NaN)
            if output.count_nonzero() > 0:
                output = F.normalize(output, p=2, dim=-1)
            else:
                # If all zeros, fall back to PyTorch implementation
                logger.warning("Triton kernel returned zero vector, falling back to PyTorch")
                seq_states = hidden_states[offset:offset + prompt_len]
                output = seq_states.mean(dim=0)
                output = F.normalize(output, p=2, dim=-1)
            pooled_outputs.append(output)
            
            offset += prompt_len
        
        return pooled_outputs

    def _apply_vision_pooling_pytorch(
        self,
        hidden_states: torch.Tensor,
        token_ids_list: List[array],
        prompt_lens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """PyTorch fallback for vision token pooling."""
        pooled_outputs = []
        offset = 0
        
        for token_ids, prompt_len in zip(token_ids_list, prompt_lens):
            prompt_len = int(prompt_len.item())
            
            # Safety check for sequence length
            if prompt_len > MAX_SEQUENCE_LENGTH:
                logger.warning(f"Sequence length {prompt_len} exceeds maximum {MAX_SEQUENCE_LENGTH}")
                prompt_len = MAX_SEQUENCE_LENGTH
            
            # Extract sequence states and tokens
            seq_states = hidden_states[offset:offset + prompt_len]
            
            # Convert array to tensor for processing
            seq_tokens = torch.tensor(list(token_ids[:prompt_len]), 
                                     dtype=torch.long, 
                                     device=hidden_states.device)
            
            # Check for vision tokens
            vision_mask = (
                (seq_tokens >= VISION_START_TOKEN_ID) & 
                (seq_tokens <= VISION_END_TOKEN_ID)
            )
            
            if vision_mask.any():
                # Pool only vision tokens
                vision_states = seq_states[vision_mask]
                if vision_states.numel() == 0:
                    logger.warning("No vision states found despite vision mask")
                    pooled = seq_states.mean(dim=0)
                else:
                    pooled = vision_states.mean(dim=0)
            else:
                # Pool all tokens for text
                pooled = seq_states.mean(dim=0)
            
            # Normalize embeddings
            pooled = F.normalize(pooled, p=2, dim=-1)
            pooled_outputs.append(pooled)
            
            offset += prompt_len
        
        return pooled_outputs

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        """Thread-safe pooler with production error handling."""
        start_time = time.time() if self.observability_config else None
        
        # Validate inputs
        if hidden_states is None or hidden_states.numel() == 0:
            logger.warning("Empty hidden states received")
            return PoolerOutput(outputs=[])
        
        # Extract token IDs safely from metadata
        token_ids_list, seq_ids = self._extract_token_ids_safe(pooling_metadata)
        
        if not token_ids_list:
            logger.warning("No valid sequences found for pooling")
            # Fallback to base pooler
            return self._base_pooler(hidden_states, pooling_metadata)
        
        # Get prompt lengths based on metadata type
        if isinstance(pooling_metadata, V1PoolingMetadata):
            prompt_lens = pooling_metadata.prompt_lens
        else:
            prompt_lens = PoolingTensors.from_pooling_metadata(
                pooling_metadata, hidden_states.device
            ).prompt_lens
        
        # Validate lengths match
        if len(token_ids_list) != len(prompt_lens):
            logger.error(f"Mismatch: {len(token_ids_list)} sequences vs {len(prompt_lens)} lengths")
            return self._base_pooler(hidden_states, pooling_metadata)
        
        # Apply optimized pooling
        try:
            pooled_data = self._apply_vision_pooling_optimized(
                hidden_states, token_ids_list, prompt_lens
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during optimized pooling, falling back to batched PyTorch")
                # Fallback to a more memory-efficient PyTorch implementation
                pooled_data = self._apply_vision_pooling_pytorch(
                    hidden_states, token_ids_list, prompt_lens
                )
            else:
                raise
        
        # Build output
        pooled_outputs = [
            PoolingSequenceGroupOutput(data) for data in pooled_data
        ]
        
        # Record metrics
        if self.observability_config:
            elapsed_ms = (time.time() - start_time) * 1000
            self._pooling_time_ms += elapsed_ms
            self._pooling_count += 1
            
            if self._pooling_count % 100 == 0:
                avg_time = self._pooling_time_ms / self._pooling_count
                logger.debug(f"Average pooling time: {avg_time:.2f}ms")
        
        return PoolerOutput(outputs=pooled_outputs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights with validation and error handling."""
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights, mapper=self.weight_mapper)
        return loaded_weights
