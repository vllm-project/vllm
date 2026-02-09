# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rate limiting for distillation capture in speculative decoding."""

import threading
from typing import Dict


class RateLimiter:
    """Thread-safe rate limiter for distillation capture.
    
    Tracks the total number of drafts processed and the number captured,
    enforcing a maximum capture percentage to control storage overhead.
    """
    
    def __init__(self, max_percentage: float):
        """Initialize rate limiter.
        
        Args:
            max_percentage: Maximum percentage of drafts to capture (0-100).
        """
        self.max_percentage = max_percentage
        self.total_drafts = 0
        self.captured_drafts = 0
        self._lock = threading.Lock()
    
    def should_capture(self) -> bool:
        """Check if we can capture another draft without exceeding the limit.
        
        This method increments the total_drafts counter and checks if
        capturing another draft would exceed the maximum capture percentage.
        
        Returns:
            True if capture is allowed, False if limit is reached.
        """
        with self._lock:
            self.total_drafts += 1
            
            # Always allow capture if we haven't captured anything yet
            if self.captured_drafts == 0:
                return True
            
            # Calculate current percentage
            current_percentage = (self.captured_drafts / self.total_drafts) * 100
            
            # Allow capture if we're below the limit
            return current_percentage < self.max_percentage
    
    # Backward compatibility alias
    def should_log(self) -> bool:
        """Check if we can capture another draft (backward compat alias)."""
        return self.should_capture()
    
    def record_captured(self) -> None:
        """Record that a draft was captured.
        
        This method should be called after successfully initiating
        capture for a draft.
        """
        with self._lock:
            self.captured_drafts += 1
    
    # Backward compatibility alias
    def record_logged(self) -> None:
        """Record that a draft was captured (backward compat alias)."""
        self.record_captured()
    
    def get_stats(self) -> Dict[str, float]:
        """Return current capture statistics.
        
        Returns:
            Dictionary with keys:
                - total_drafts: Total number of drafts processed
                - captured_drafts: Number of drafts captured
                - percentage: Current capture percentage
        """
        with self._lock:
            percentage = (
                (self.captured_drafts / self.total_drafts * 100)
                if self.total_drafts > 0
                else 0.0
            )
            
            return {
                "total_drafts": self.total_drafts,
                "captured_drafts": self.captured_drafts,
                "percentage": percentage,
            }
