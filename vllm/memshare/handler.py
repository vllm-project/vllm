"""
MemShare handler — per-request state and step detection for KV cache sharing.

Hooks into vLLM's output processor to detect reasoning step boundaries
and find candidates for block sharing.

This is the integration point between vLLM's decode loop and the
MemShare pipeline: step detection → Stage 1 cosine similarity →
(future) Stage 2 KV distance → block remap.
"""

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Step boundary patterns — derived from manual annotation of AIME 2024 traces
BOUNDARY_PATTERNS = [
    r"Wait,?\s",
    r"Hmm,?\s",
    r"Let me verify",
    r"Let me check",
    r"Let me reconsider",
    r"Let me re-?compute",
    r"Let me re-?calculate",
    r"Let me see",
    r"Let me think",
    r"Let me try",
    r"Another thought:",
    r"Another idea:",
    r"Another approach:",
    r"Actually,?\s",
    r"But wait,?\s",
    r"But let me",
    r"Perhaps I should",
    r"Is that correct\??",
    r"Is that true\??",
    r"Is that right\??",
    r"I recall that",
    r"I think I recall",
    r"I remember that",
]

_BOUNDARY_RE = re.compile(
    r"(?:^|\n|[.!?]\s)\s*(" + "|".join(BOUNDARY_PATTERNS) + r")",
    re.IGNORECASE,
)

MIN_STEP_CHARS = 40


class RequestMemShareState:
    """Per-request state for MemShare step detection and similarity."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.text_buffer: str = ""
        self.completed_steps: List[str] = []
        self.step_vectors: List[Counter] = []
        self.candidates: List[Tuple[int, int, float]] = []

    def _tokenize(self, text: str) -> Counter:
        return Counter(text.lower().split())

    def _cosine_similarity(self, a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def on_new_text(self, new_text: str, threshold: float = 0.8) -> None:
        """Called each time new decoded text is available for this request.

        Appends to the buffer, checks for step boundaries, and runs
        Stage 1 similarity on any completed steps.
        """
        self.text_buffer += new_text

        # Check if the buffer tail contains a boundary pattern
        match = _BOUNDARY_RE.search(self.text_buffer)
        if match and match.start(1) >= MIN_STEP_CHARS:
            split_pos = match.start(1)
            completed = self.text_buffer[:split_pos].strip()

            if completed:
                self._complete_step(completed, threshold)

            # Keep the boundary phrase in the buffer as the start of the next step
            self.text_buffer = self.text_buffer[split_pos:]

    def _complete_step(self, step_text: str, threshold: float) -> None:
        """Register a completed step and compare against previous steps."""
        step_idx = len(self.completed_steps)
        self.completed_steps.append(step_text)

        vec = self._tokenize(step_text)
        self.step_vectors.append(vec)

        # Compare against all previous steps
        for prev_idx, prev_vec in enumerate(self.step_vectors[:-1]):
            sim = self._cosine_similarity(vec, prev_vec)
            if sim >= threshold:
                self.candidates.append((prev_idx, step_idx, sim))
                logger.info(
                    "MemShare [%s]: step %d ≈ step %d (sim=%.3f)",
                    self.request_id, prev_idx, step_idx, sim,
                )

    def finalize(self) -> None:
        """Called when the request finishes. Completes the final step."""
        remaining = self.text_buffer.strip()
        if remaining:
            self._complete_step(remaining, threshold=0.8)
            self.text_buffer = ""


class MemShareHandler:
    """Manages per-request MemShare state across all active requests."""

    def __init__(self, enabled: bool = True, threshold: float = 0.8):
        self.enabled = enabled
        self.threshold = threshold
        self._states: Dict[str, RequestMemShareState] = {}

    def on_token(self, request_id: str, new_text: str) -> None:
        """Called after each token is decoded to text."""
        if not self.enabled or not new_text:
            return

        if request_id not in self._states:
            self._states[request_id] = RequestMemShareState(request_id)

        self._states[request_id].on_new_text(new_text, self.threshold)

    def on_request_finished(self, request_id: str) -> Optional[RequestMemShareState]:
        """Called when a request completes. Returns the final state."""
        state = self._states.pop(request_id, None)
        if state:
            state.finalize()
            logger.info(
                "MemShare [%s]: %d steps, %d candidates",
                request_id, len(state.completed_steps), len(state.candidates),
            )
        return state
