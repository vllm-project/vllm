# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Data collector for speculative decoding training (EAGLE3-style).

This module provides infrastructure to collect training data for speculative
decoding models by capturing:
- Input tokens
- Hidden states from arbitrary model layers
- Output logits

The collected data can be used to train EAGLE-style draft models.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for data collection."""

    # Whether collection is enabled
    enabled: bool = False

    # Directory to save collected data
    output_dir: str = "/tmp/vllm_spec_decode_data"

    # Which layers to collect hidden states from (empty = all layers)
    # Format: list of layer indices, e.g., [0, 8, 16, 24, 31]
    hidden_state_layers: list[int] = field(default_factory=list)

    # Maximum number of samples to collect per file
    samples_per_file: int = 1000

    # Maximum buffer size before flushing to disk
    max_buffer_size: int = 100

    # Whether to collect logits
    collect_logits: bool = True

    # Whether to collect hidden states
    collect_hidden_states: bool = True

    # Worker rank for multi-worker setups (e.g., tensor parallel)
    # Set to None to disable rank-based file naming
    worker_rank: int | None = None


@dataclass
class DataSample:
    """A single training sample for spec decoding."""

    # Request ID
    request_id: str

    # Input token IDs
    token_ids: list[int]

    # Hidden states from specified layers
    # Dict mapping layer_idx -> tensor (on CPU, converted to numpy)
    hidden_states: dict[int, Any] = field(default_factory=dict)

    # Output logits (on CPU, converted to numpy)
    logits: Any = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    num_tokens: int = 0


class SpecDecodeDataCollector:
    """
    Collects training data for speculative decoding models.

    This collector can be enabled/disabled at runtime via an HTTP endpoint.
    It efficiently batches data in memory and periodically flushes to disk.
    """

    def __init__(self, config: CollectionConfig | None = None):
        self.config = config or CollectionConfig()
        self._buffer: list[DataSample] = []
        self._file_counter = 0
        self._sample_counter = 0
        self._lock = threading.Lock()
        self._registered_hooks: list[Any] = []
        self._layer_hidden_states: dict[int, torch.Tensor] = {}

        # Create output directory if it doesn't exist
        if self.config.enabled:
            self._ensure_output_dir()
            logger.info(
                "SpecDecodeDataCollector initialized: output_dir=%s, "
                "layers=%s, samples_per_file=%d",
                self.config.output_dir,
                self.config.hidden_state_layers
                if self.config.hidden_state_layers
                else "all",
                self.config.samples_per_file,
            )

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def is_enabled(self) -> bool:
        """Check if data collection is currently enabled."""
        return self.config.enabled

    def enable(
        self,
        output_dir: str | None = None,
        model: nn.Module | None = None,
        **kwargs,
    ):
        """
        Enable data collection.

        Args:
            output_dir: Optional output directory (uses config default if None)
            model: Optional model to register hooks on
            **kwargs: Additional config overrides
        """
        with self._lock:
            # Clear any existing hooks before registering new ones
            self.unregister_layer_hooks()

            self.config.enabled = True
            if output_dir:
                self.config.output_dir = output_dir
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            self._ensure_output_dir()
            logger.info(
                "Data collection enabled: output_dir=%s", self.config.output_dir
            )

            # Register hooks if model is provided
            if model is not None and self.config.collect_hidden_states:
                self.register_layer_hooks(model)

    def disable(self):
        """Disable data collection and flush any remaining data."""
        with self._lock:
            if self.config.enabled:
                self.config.enabled = False
                self._flush_buffer()
                self.unregister_layer_hooks()
                logger.info("Data collection disabled")

    def collect_step(
        self,
        request_ids: list[str],
        token_ids_list: list[list[int]],
        hidden_states: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        layer_hidden_states: dict[int, torch.Tensor] | None = None,
    ):
        """
        Collect data from a single forward pass.

        Args:
            request_ids: List of request IDs in the batch
            token_ids_list: List of token ID lists for each request
            hidden_states: Final hidden states [num_tokens, hidden_size]
            logits: Output logits [num_tokens, vocab_size]
            layer_hidden_states: Dict of layer_idx -> hidden states
                (if None, will use cached states from forward hooks)
        """
        if not self.config.enabled:
            return

        try:
            with self._lock:
                # Use provided layer_hidden_states or fall back to cached states
                states_to_use = (
                    layer_hidden_states
                    if layer_hidden_states is not None
                    else self._layer_hidden_states
                )

                for i, (req_id, token_ids) in enumerate(
                    zip(request_ids, token_ids_list)
                ):
                    # Create sample with a defensive copy of token_ids
                    # to prevent mutation by subsequent generation steps
                    sample = DataSample(
                        request_id=req_id,
                        token_ids=list(token_ids),
                        num_tokens=len(token_ids),
                    )

                    # Collect hidden states from specified layers
                    if self.config.collect_hidden_states and states_to_use is not None:
                        for layer_idx, layer_hidden in states_to_use.items():
                            # Skip if not in requested layers
                            # (empty = collect all)
                            if (
                                self.config.hidden_state_layers
                                and layer_idx not in self.config.hidden_state_layers
                            ):
                                continue

                            # Extract only this sample's hidden states (index by i)
                            # Convert bf16 to fp32 if needed (numpy doesn't
                            # support bf16)
                            if isinstance(layer_hidden, torch.Tensor):
                                tensor = layer_hidden[i].detach().cpu()
                                if tensor.dtype == torch.bfloat16:
                                    tensor = tensor.to(torch.float32)
                                sample.hidden_states[layer_idx] = tensor.numpy()

                    # Collect logits
                    # Only save logits for this sample
                    # Convert to float32 first if bfloat16
                    # (numpy doesn't support bf16)
                    if (
                        self.config.collect_logits
                        and logits is not None
                        and isinstance(logits, torch.Tensor)
                    ):
                        logit_tensor = logits[i].detach().cpu()
                        if logit_tensor.dtype == torch.bfloat16:
                            logit_tensor = logit_tensor.to(torch.float32)
                        sample.logits = logit_tensor.numpy()

                    self._buffer.append(sample)
                    self._sample_counter += 1

                # Flush if buffer is full
                if len(self._buffer) >= self.config.max_buffer_size:
                    self._flush_buffer()

                # Clear cached layer states for next iteration
                self.clear_layer_hidden_states()

        except Exception as e:
            logger.exception("Error collecting data: %s", e)

    def _flush_buffer(self):
        """Flush buffered samples to disk."""
        if not self._buffer:
            return

        try:
            # Group samples into files
            while self._buffer:
                chunk_size = min(len(self._buffer), self.config.samples_per_file)
                chunk = self._buffer[:chunk_size]
                self._buffer = self._buffer[chunk_size:]

                # Save to file with optional worker rank
                if self.config.worker_rank is not None:
                    filename = (
                        f"spec_decode_data_rank{self.config.worker_rank}_"
                        f"{self._file_counter:06d}.npz"
                    )
                else:
                    filename = f"spec_decode_data_{self._file_counter:06d}.npz"
                filepath = os.path.join(self.config.output_dir, filename)

                # Prepare data for saving
                data_dict: dict[str, Any] = {
                    "request_ids": [s.request_id for s in chunk],
                    "timestamps": np.array([s.timestamp for s in chunk]),
                    "num_tokens": np.array([s.num_tokens for s in chunk]),
                }

                # Save token IDs (variable length, so save as object array)
                max_len = max(len(s.token_ids) for s in chunk)
                token_ids_padded = np.zeros((len(chunk), max_len), dtype=np.int32)
                token_lens = np.zeros(len(chunk), dtype=np.int32)
                for i, sample in enumerate(chunk):
                    token_ids_padded[i, : len(sample.token_ids)] = sample.token_ids
                    token_lens[i] = len(sample.token_ids)
                data_dict["token_ids"] = token_ids_padded
                data_dict["token_lens"] = token_lens

                # Save hidden states by layer
                if chunk[0].hidden_states:
                    layer_indices = sorted(chunk[0].hidden_states.keys())
                    data_dict["layer_indices"] = np.array(layer_indices)
                    for layer_idx in layer_indices:
                        hidden_list = [s.hidden_states.get(layer_idx) for s in chunk]
                        if all(h is not None for h in hidden_list):
                            data_dict[f"hidden_states_layer_{layer_idx}"] = np.stack(
                                hidden_list
                            )

                # Save logits
                if chunk[0].logits is not None:
                    logits_list = [s.logits for s in chunk if s.logits is not None]
                    if logits_list:
                        data_dict["logits"] = np.stack(logits_list)

                # Save to npz file
                np.savez_compressed(filepath, **data_dict)

                # Also save metadata as JSON for easy inspection
                metadata_file = filepath.replace(".npz", "_metadata.json")
                metadata = {
                    "num_samples": len(chunk),
                    "layer_indices": (layer_indices if chunk[0].hidden_states else []),
                    "has_logits": chunk[0].logits is not None,
                    "timestamp_range": [
                        min(s.timestamp for s in chunk),
                        max(s.timestamp for s in chunk),
                    ],
                }
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info("Saved %d samples to %s", len(chunk), filename)
                self._file_counter += 1

        except Exception as e:
            logger.exception("Error flushing buffer: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        with self._lock:
            return {
                "enabled": self.config.enabled,
                "total_samples_collected": self._sample_counter,
                "files_written": self._file_counter,
                "buffer_size": len(self._buffer),
                "output_dir": self.config.output_dir,
                "config": asdict(self.config),
            }

    def register_layer_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks to capture hidden states from specified layers.

        Args:
            model: The PyTorch model to register hooks on
        """
        if not self.config.enabled or not self.config.collect_hidden_states:
            return

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                if self.config.enabled:
                    # Store the output hidden states for this layer
                    # Handle both tuple outputs and tensor outputs
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    self._layer_hidden_states[layer_idx] = hidden_states

            return hook

        # Register hooks for specified layers
        # This assumes the model has a structure like model.layers[i]
        if hasattr(model, "layers"):
            layers_to_hook = (
                self.config.hidden_state_layers
                if self.config.hidden_state_layers
                else range(len(model.layers))
            )
            for layer_idx in layers_to_hook:
                if layer_idx < len(model.layers):
                    handle = model.layers[layer_idx].register_forward_hook(
                        make_hook(layer_idx)
                    )
                    self._registered_hooks.append(handle)
            logger.info(
                "Registered forward hooks for %d layers", len(self._registered_hooks)
            )
        else:
            logger.warning(
                "Model does not have 'layers' attribute. "
                "Cannot register layer hooks automatically."
            )

    def unregister_layer_hooks(self) -> None:
        """Unregister all forward hooks."""
        for handle in self._registered_hooks:
            handle.remove()
        self._registered_hooks.clear()
        logger.info("Unregistered all layer hooks")

    def clear_layer_hidden_states(self) -> None:
        """Clear cached layer hidden states."""
        self._layer_hidden_states.clear()

    def shutdown(self):
        """Shutdown the collector and flush all data."""
        with self._lock:
            logger.info("Shutting down data collector")
            self.unregister_layer_hooks()
            self._flush_buffer()
            logger.info(
                "Data collector shutdown complete. Total samples: %d, Files: %d",
                self._sample_counter,
                self._file_counter,
            )


# Global collector instance
_global_collector: SpecDecodeDataCollector | None = None


def get_global_collector() -> SpecDecodeDataCollector:
    """Get or create the global data collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = SpecDecodeDataCollector()
    return _global_collector


def initialize_collector(config: CollectionConfig | None = None):
    """Initialize the global collector with a specific config."""
    global _global_collector
    _global_collector = SpecDecodeDataCollector(config)
    return _global_collector
