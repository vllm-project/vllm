# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Percentile-based tracking for worst acceptance rates."""

import threading
from collections import deque
from typing import Optional

import numpy as np


class PercentileTracker:
    """Tracks acceptance rates and determines worst X percentile.
    
    Maintains a sliding window of recent acceptance rates to dynamically
    determine which drafts are in the worst percentile. This allows logging
    only the most problematic cases for knowledge distillation.
    """
    
    def __init__(
        self,
        percentile: float = 10.0,
        window_size: int = 1000,
        min_samples: int = 100,
    ):
        """Initialize percentile tracker.
        
        Args:
            percentile: Percentile threshold (0-100). Log drafts in worst X%.
            window_size: Number of recent acceptance rates to track.
            min_samples: Minimum samples before percentile is reliable.
        """
        self.percentile = percentile
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Sliding window of recent acceptance lengths
        self.acceptance_window: deque[float] = deque(maxlen=window_size)
        
        # Cached percentile threshold (updated periodically)
        self._cached_threshold: Optional[float] = None
        self._cache_valid = False
        
        self._lock = threading.Lock()
    
    def observe(self, acceptance_length: float) -> None:
        """Record a new acceptance length observation.
        
        Args:
            acceptance_length: Acceptance length for this draft.
        """
        with self._lock:
            self.acceptance_window.append(acceptance_length)
            # Invalidate cache when window changes significantly
            if len(self.acceptance_window) % 10 == 0:
                self._cache_valid = False
    
    def should_capture(self, acceptance_length: float) -> bool:
        """Check if this acceptance length is in the worst percentile.
        
        Args:
            acceptance_length: Acceptance length to check.
        
        Returns:
            True if in worst percentile and should be captured.
        """
        with self._lock:
            # Need minimum samples for reliable percentile
            if len(self.acceptance_window) < self.min_samples:
                # During warmup, capture everything below a conservative threshold
                return bool(acceptance_length < 1.5)
            
            # Update cached threshold if needed
            if not self._cache_valid:
                self._update_threshold()
            
            # Capture if below the percentile threshold
            result = acceptance_length <= self._cached_threshold
            return bool(result.item() if hasattr(result, 'item') else result)
    
    # Backward compatibility alias
    def should_log(self, acceptance_length: float) -> bool:
        """Check if should capture (backward compat alias)."""
        return self.should_capture(acceptance_length)
    
    def observe_and_check_capture(self, acceptance_length: float) -> bool:
        """Atomically observe and check if should capture.
        
        This method combines observe() and should_capture() into a single
        atomic operation, ensuring the decision is based on the most
        recent data including this observation.
        
        Optimized to minimize lock contention by using fast path when
        cache is valid.
        
        Args:
            acceptance_length: Acceptance length for this draft.
        
        Returns:
            True if in worst percentile and should be captured.
        """
        # Fast path: check without lock if cache is valid and we have enough samples
        # This reduces lock contention in high-throughput scenarios
        if self._cache_valid and len(self.acceptance_window) >= self.min_samples:
            # Read cached threshold (atomic read, no lock needed)
            cached_threshold = self._cached_threshold
            result = acceptance_length <= cached_threshold
            should_capture_result = bool(result.item() if hasattr(result, 'item') else result)
            
            # Quick lock just to append observation
            with self._lock:
                self.acceptance_window.append(acceptance_length)
                # Invalidate cache periodically
                if len(self.acceptance_window) % 10 == 0:
                    self._cache_valid = False
            
            return should_capture_result
        
        # Slow path: need to compute or update threshold
        with self._lock:
            # Check before adding (decision based on historical data)
            should_capture_result = False
            
            if len(self.acceptance_window) < self.min_samples:
                # During warmup, capture everything below a conservative threshold
                should_capture_result = bool(acceptance_length < 1.5)
            else:
                # Update cached threshold if needed
                if not self._cache_valid:
                    self._update_threshold()
                
                # Capture if below the percentile threshold
                result = acceptance_length <= self._cached_threshold
                should_capture_result = bool(result.item() if hasattr(result, 'item') else result)
            
            # Now add the observation
            self.acceptance_window.append(acceptance_length)
            # Invalidate cache when window changes significantly
            if len(self.acceptance_window) % 10 == 0:
                self._cache_valid = False
            
            return should_capture_result
    
    # Backward compatibility alias
    def observe_and_check(self, acceptance_length: float) -> bool:
        """Atomically observe and check if should capture (backward compat alias)."""
        return self.observe_and_check_capture(acceptance_length)
    
    def _update_threshold(self) -> None:
        """Update the cached percentile threshold.
        
        This is called periodically to avoid recomputing percentile
        on every check.
        """
        if len(self.acceptance_window) == 0:
            self._cached_threshold = float('inf')
            return
        
        # Compute the percentile threshold
        acceptance_array = np.array(self.acceptance_window)
        self._cached_threshold = np.percentile(acceptance_array, self.percentile)
        self._cache_valid = True
    
    def get_stats(self) -> dict:
        """Get current tracking statistics.
        
        Returns:
            Dictionary with statistics about tracked acceptance rates.
        """
        with self._lock:
            if len(self.acceptance_window) == 0:
                return {
                    "num_samples": 0,
                    "percentile_threshold": None,
                    "mean_acceptance": None,
                    "min_acceptance": None,
                    "max_acceptance": None,
                }
            
            acceptance_array = np.array(self.acceptance_window)
            
            # Update threshold if needed
            if not self._cache_valid:
                self._update_threshold()
            
            return {
                "num_samples": len(self.acceptance_window),
                "percentile_threshold": self._cached_threshold,
                "mean_acceptance": float(np.mean(acceptance_array)),
                "min_acceptance": float(np.min(acceptance_array)),
                "max_acceptance": float(np.max(acceptance_array)),
                "p25": float(np.percentile(acceptance_array, 25)),
                "p50": float(np.percentile(acceptance_array, 50)),
                "p75": float(np.percentile(acceptance_array, 75)),
            }
    
    def reset(self) -> None:
        """Reset the tracker, clearing all observations."""
        with self._lock:
            self.acceptance_window.clear()
            self._cached_threshold = None
            self._cache_valid = False
