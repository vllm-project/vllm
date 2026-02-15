# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asynchronous Safetensors writer for distillation capture.

Stores logits and hidden states using safetensors format with optional
zstd compression. Preserves native tensor dtypes (fp16, bf16, fp8, int8, etc.)
without conversion.
"""

import json
import os
import queue
import tempfile
import threading
import time
from collections import deque
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from vllm.logger import init_logger

logger = init_logger(__name__)

# Optional zstd support
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard not installed. Compression disabled.")


class AsyncSafetensorsWriter:
    """Asynchronous Safetensors writer with bounded queue.
    
    Handles background file I/O operations for writing logits data to
    Safetensors files. Preserves tensor dtypes natively and supports
    optional zstd compression.
    
    Uses a staging queue with CUDA events to ensure GPU-to-CPU transfers
    complete before accessing tensors.
    """
    
    def __init__(
        self,
        output_dir: str,
        queue_size: int,
        batch_size: int = 10,
        batch_timeout: float = 5.0,
        use_compression: bool = True,
        compression_level: int = 3,
    ):
        """Initialize async Safetensors writer.
        
        Args:
            output_dir: Directory where files will be written.
            queue_size: Maximum number of pending write operations.
            batch_size: Number of samples to batch before writing.
            batch_timeout: Max seconds to wait for batch to fill.
            use_compression: Whether to apply zstd compression.
            compression_level: Zstd compression level (1-22, default 3).
        """
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.use_compression = use_compression and ZSTD_AVAILABLE
        self.compression_level = compression_level
        
        if use_compression and not ZSTD_AVAILABLE:
            logger.warning(
                "Compression requested but zstandard not installed. "
                "Install with: pip install zstandard"
            )
        
        # Queues
        self.staging_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.ready_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._shutdown = False
        
        # Adaptive polling
        self._transfer_times: deque[float] = deque(maxlen=100)
        self._min_poll_interval = 0.0001
        self._max_poll_interval = 0.01
        self._current_poll_interval = 0.001
        
        # Statistics
        self._total_drops = 0
        self._total_writes = 0
        self._total_batches = 0
        self._total_bytes = 0
        self._lock = threading.Lock()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(
            f"Safetensors writer initialized: dir={output_dir}, "
            f"compression={self.use_compression}"
        )
        
        # Start background threads
        self.checker_thread = threading.Thread(
            target=self._transfer_checker_loop,
            daemon=True,
            name="distillation-transfer-checker"
        )
        self.checker_thread.start()
        
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="distillation-safetensors-writer"
        )
        self.writer_thread.start()
    
    def queue_write(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
        input_ids: torch.Tensor,
        metadata: Dict,
        cuda_event: Optional[torch.cuda.Event] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Queue data for writing.
        
        Non-blocking if queue has space. Drops data if queue is full.
        
        Args:
            probs: Top-k probabilities tensor [batch_size, seq_len, k].
            indices: Top-k token indices tensor [batch_size, seq_len, k].
            input_ids: Input token IDs tensor [batch_size, seq_len].
            metadata: Dictionary with acceptance_length, timestamp, etc.
            cuda_event: CUDA event to synchronize before accessing tensors.
            hidden_states: Hidden states tensor (optional).
        """
        try:
            self.staging_queue.put_nowait({
                'probs': probs,
                'indices': indices,
                'input_ids': input_ids,
                'metadata': metadata,
                'cuda_event': cuda_event,
                'hidden_states': hidden_states,
            })
        except queue.Full:
            with self._lock:
                self._total_drops += 1
            logger.warning(
                f"Staging queue full, dropping data. Total drops: {self._total_drops}"
            )
    
    def _transfer_checker_loop(self) -> None:
        """Check if CUDA transfers are complete and move to ready queue."""
        pending_items: list[tuple[Dict, float]] = []
        
        while not self._shutdown:
            try:
                # Get new items
                try:
                    data = self.staging_queue.get_nowait()
                    pending_items.append((data, time.time()))
                except queue.Empty:
                    pass
                
                # Check pending items
                still_pending = []
                for data, enqueue_time in pending_items:
                    cuda_event = data.get('cuda_event')
                    
                    if cuda_event is not None:
                        if cuda_event.query():
                            transfer_time = time.time() - enqueue_time
                            self._update_transfer_stats(transfer_time)
                            try:
                                self.ready_queue.put_nowait(data)
                            except queue.Full:
                                with self._lock:
                                    self._total_drops += 1
                        else:
                            still_pending.append((data, enqueue_time))
                    else:
                        try:
                            self.ready_queue.put_nowait(data)
                        except queue.Full:
                            with self._lock:
                                self._total_drops += 1
                
                pending_items = still_pending
                
                if not pending_items:
                    time.sleep(self._max_poll_interval)
                else:
                    time.sleep(self._current_poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in transfer checker: {e}")
    
    def _update_transfer_stats(self, transfer_time: float) -> None:
        """Update transfer statistics and adjust polling."""
        self._transfer_times.append(transfer_time)
        
        if len(self._transfer_times) >= 10:
            median_time = sorted(self._transfer_times)[len(self._transfer_times) // 2]
            optimal_interval = median_time / 4.0
            self._current_poll_interval = max(
                self._min_poll_interval,
                min(self._max_poll_interval, optimal_interval)
            )
    
    def _writer_loop(self) -> None:
        """Process write queue and write batches to files."""
        batch: list[Dict] = []
        last_write_time = time.time()
        
        while not self._shutdown:
            try:
                try:
                    data = self.ready_queue.get(timeout=0.1)
                    batch.append(data)
                except queue.Empty:
                    pass
                
                should_write = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and time.time() - last_write_time >= self.batch_timeout) or
                    (self._shutdown and len(batch) > 0)
                )
                
                if should_write:
                    self._write_batch(batch)
                    batch.clear()
                    last_write_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in writer loop: {e}")
                batch.clear()
    
    def _write_batch(self, batch: list[Dict]) -> None:
        """Write a batch of data to a safetensors file."""
        try:
            # Collect tensors from batch
            all_probs = []
            all_indices = []
            all_input_ids = []
            all_hidden_states = []
            all_metadata = []
            
            for data in batch:
                probs = data['probs']
                indices = data['indices']
                input_ids = data['input_ids']
                
                # Ensure on CPU
                if probs.device.type != 'cpu':
                    probs = probs.cpu()
                if indices.device.type != 'cpu':
                    indices = indices.cpu()
                if input_ids.device.type != 'cpu':
                    input_ids = input_ids.cpu()
                
                all_probs.append(probs)
                all_indices.append(indices)
                all_input_ids.append(input_ids)
                all_metadata.append(data['metadata'])
                
                if data.get('hidden_states') is not None:
                    hs = data['hidden_states']
                    if hs.device.type != 'cpu':
                        hs = hs.cpu()
                    all_hidden_states.append(hs)
            
            # Concatenate tensors
            tensors = {
                'probs': torch.cat(all_probs, dim=0),
                'indices': torch.cat(all_indices, dim=0),
                'input_ids': torch.cat(all_input_ids, dim=0),
            }
            
            if all_hidden_states:
                tensors['hidden_states'] = torch.cat(all_hidden_states, dim=0)
            
            # Store metadata as JSON in safetensors metadata field
            metadata_json = json.dumps({
                'batch_size': len(batch),
                'metadata': all_metadata,
                'timestamp': time.time(),
            })
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            if self.use_compression:
                filename = f"capture_{timestamp}.safetensors.zst"
            else:
                filename = f"capture_{timestamp}.safetensors"
            filepath = os.path.join(self.output_dir, filename)
            
            # Write file
            self._atomic_write(tensors, filepath, metadata_json)
            
            # Update stats
            file_size = os.path.getsize(filepath)
            with self._lock:
                self._total_writes += len(batch)
                self._total_batches += 1
                self._total_bytes += file_size
            
            logger.debug(
                f"Wrote {len(batch)} samples to {filename} "
                f"({file_size / 1024:.1f} KB)"
            )
            
        except Exception as e:
            logger.error(f"Error writing batch: {e}")
    
    def _atomic_write(
        self,
        tensors: Dict[str, torch.Tensor],
        filepath: str,
        metadata_json: str,
    ) -> None:
        """Atomically write tensors to file with optional compression."""
        fd, tmp_path = tempfile.mkstemp(
            dir=self.output_dir,
            prefix="tmp_safetensors_"
        )
        os.close(fd)
        
        try:
            if self.use_compression:
                # Write to temp safetensors first
                tmp_st = tmp_path + ".st"
                save_file(tensors, tmp_st, metadata={'metadata': metadata_json})
                
                # Compress with zstd
                cctx = zstd.ZstdCompressor(level=self.compression_level)
                with open(tmp_st, 'rb') as f_in:
                    with open(tmp_path, 'wb') as f_out:
                        cctx.copy_stream(f_in, f_out)
                
                # Clean up uncompressed temp
                os.remove(tmp_st)
            else:
                save_file(tensors, tmp_path, metadata={'metadata': metadata_json})
            
            os.replace(tmp_path, filepath)
            
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    
    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown writer threads."""
        logger.info(
            f"Shutting down safetensors writer. "
            f"Writes: {self._total_writes}, Batches: {self._total_batches}, "
            f"Drops: {self._total_drops}, Bytes: {self._total_bytes / 1024 / 1024:.1f} MB"
        )
        
        self._shutdown = True
        self.checker_thread.join(timeout=timeout)
        self.writer_thread.join(timeout=timeout)
        
        if self.checker_thread.is_alive():
            logger.warning("Checker thread did not shutdown cleanly")
        if self.writer_thread.is_alive():
            logger.warning("Writer thread did not shutdown cleanly")
    
    def get_stats(self) -> dict:
        """Get current writer statistics."""
        with self._lock:
            return {
                'staging_queue_size': self.staging_queue.qsize(),
                'ready_queue_size': self.ready_queue.qsize(),
                'total_writes': self._total_writes,
                'total_batches': self._total_batches,
                'total_drops': self._total_drops,
                'total_bytes': self._total_bytes,
                'use_compression': self.use_compression,
            }
