"""
Harmony utilities for GPT-OSS model support.
"""
import os
from typing import Optional

try:
    from openai_harmony import load_harmony_encoding
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

# Global harmony encoding instance
_harmony_encoding = None


def is_harmony_available() -> bool:
    """Check if openai-harmony is available."""
    return HARMONY_AVAILABLE


def get_encoding(name: str = "o200k_harmony") -> Optional[object]:
    """Get the harmony encoding instance."""
    global _harmony_encoding
    
    if not HARMONY_AVAILABLE:
        return None
        
    if _harmony_encoding is None:
        try:
            _harmony_encoding = load_harmony_encoding(name)
        except Exception as e:
            # Handle cases where harmony vocab might not be available
            # in air-gapped environments
            print(f"Warning: Could not load harmony encoding: {e}")
            return None
    
    return _harmony_encoding


def get_stop_tokens_for_assistant_actions():
    """Get stop tokens for assistant actions."""
    encoding = get_encoding()
    if encoding is None:
        return []
    
    try:
        return encoding.stop_tokens_for_assistant_actions()
    except AttributeError:
        # Fallback if method doesn't exist
        return []


def encode_reasoning_token(token_type: str = "reasoning"):
    """Encode reasoning tokens for GPT-OSS."""
    encoding = get_encoding()
    if encoding is None:
        return []
    
    try:
        # This is a placeholder - actual implementation depends on harmony API
        return encoding.encode(f"<|{token_type}|>")
    except Exception:
        return []


def is_reasoning_token(token_id: int) -> bool:
    """Check if a token ID represents a reasoning token."""
    encoding = get_encoding()
    if encoding is None:
        return False
    
    try:
        # This is a placeholder - actual implementation depends on harmony API
        decoded = encoding.decode([token_id])
        return decoded.startswith("<|") and decoded.endswith("|>")
    except Exception:
        return False
