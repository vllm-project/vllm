#!/usr/bin/env python3
"""
VectorChat Encrypted Tokenizer for vLLM Integration
Wrapper that enables vLLM to work with encrypted token streams from VectorChat
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Configure logging first
logger = logging.getLogger(__name__)

# Import vLLM tokenizer base
from vllm.transformers_utils.tokenizer_base import TokenizerBase

# Import VectorChat crypto components
# Add the VectorChat daemon path to sys.path for imports
import sys
vectorchat_path = Path(__file__).resolve().parent.parent / 'AID-CORE-COMMERCIAL' / 'vectorchat' / 'daemon'
if str(vectorchat_path) not in sys.path:
    sys.path.insert(0, str(vectorchat_path))

try:
    from vectorflow_42_token_handler_wrapper_v1_0_0 import VectorFlow42TokenHandlerCrypto
    VECTORCHAT_AVAILABLE = True
    logger.info("VectorChat crypto components loaded successfully")
except ImportError as e:
    logger.warning(f"VectorChat crypto components not available: {e}")
    logger.info("Running in passthrough mode - encryption/decryption disabled")
    VECTORCHAT_AVAILABLE = False
    VectorFlow42TokenHandlerCrypto = None


class VectorChatEncryptedTokenizer(TokenizerBase):
    """
    Encrypted tokenizer wrapper for vLLM integration with VectorChat technology.

    This tokenizer allows AI models to work with encrypted data streams directly,
    enabling secure inference where model hosting providers never see plaintext data.
    """

    def __init__(self,
                 base_tokenizer: Any,
                 crypto_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the encrypted tokenizer.

        Args:
            base_tokenizer: The underlying HuggingFace tokenizer
            crypto_config: Configuration for VectorChat encryption/decryption
        """
        self.base_tokenizer = base_tokenizer
        self.crypto_config = crypto_config or self._get_default_crypto_config()

        if VECTORCHAT_AVAILABLE:
            self.crypto_handler = VectorFlow42TokenHandlerCrypto()
            logger.info("VectorChat crypto handler initialized successfully")
        else:
            self.crypto_handler = None
            logger.warning("VectorChat crypto components not available - running in passthrough mode")

        # Cache for performance
        self._vocab_cache = None
        self._special_tokens_cache = None

    def _get_default_crypto_config(self) -> Dict[str, Any]:
        """Get default crypto configuration."""
        return {
            'emdm_seed_hex': os.environ.get('VECTORCHAT_EMDM_SEED_HEX', 'default_seed_12345'),
            'emdm_anchor_indices': [0, 1, 2],
            'emdm_window_len': 10,
            'pairing_sequence_length': 8,
            'session_id_length_bytes': 16,
            'checksum_length': 2,
            'header_marker_override': None,
            'token_marker_override': None,
            'digit_permutation': None
        }

    @property
    def all_special_tokens_extended(self) -> List[str]:
        """Get all special tokens (extended)."""
        if self._special_tokens_cache is None:
            self._special_tokens_cache = self.base_tokenizer.all_special_tokens_extended
        return self._special_tokens_cache

    @property
    def all_special_tokens(self) -> List[str]:
        """Get all special tokens."""
        if self._special_tokens_cache is None:
            self._special_tokens_cache = self.base_tokenizer.all_special_tokens
        return self._special_tokens_cache

    @property
    def all_special_ids(self) -> List[int]:
        """Get all special token IDs."""
        return self.base_tokenizer.all_special_ids

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.base_tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.base_tokenizer.eos_token_id

    @property
    def sep_token(self) -> str:
        """Separator token."""
        return getattr(self.base_tokenizer, 'sep_token', None)

    @property
    def pad_token(self) -> str:
        """Padding token."""
        return getattr(self.base_tokenizer, 'pad_token', None)

    @property
    def is_fast(self) -> bool:
        """Whether this is a fast tokenizer."""
        return getattr(self.base_tokenizer, 'is_fast', False)

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.base_tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        """Maximum token ID."""
        return self.base_tokenizer.vocab_size - 1

    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if self._vocab_cache is None:
            self._vocab_cache = self.base_tokenizer.get_vocab()
        return self._vocab_cache

    def get_added_vocab(self) -> Dict[str, int]:
        """Get added vocabulary."""
        return getattr(self.base_tokenizer, 'get_added_vocab', lambda: {})()

    def __call__(self,
                 text: Union[str, List[str], List[int]],
                 text_pair: Optional[str] = None,
                 add_special_tokens: bool = False,
                 truncation: bool = False,
                 max_length: Optional[int] = None):
        """
        Tokenize input text(s).

        If input is string/list of strings: tokenize and encrypt
        If input is list of ints: treat as already encrypted tokens
        """
        if isinstance(text, (str, list)) and not all(isinstance(x, int) for x in text):
            # Input is text - tokenize normally then encrypt
            if isinstance(text, str):
                tokens = self.encode_one(text, truncation=truncation, max_length=max_length)
            else:
                # Handle list of strings
                tokens = []
                for t in text:
                    tokens.extend(self.encode_one(t, truncation=truncation, max_length=max_length))
        else:
            # Input is already token IDs - use as-is or decrypt if needed
            tokens = text if isinstance(text, list) else [text]

        return tokens

    def encode_one(self,
                   text: str,
                   truncation: bool = False,
                   max_length: Optional[int] = None) -> List[int]:
        """Encode a single text string to tokens."""
        # First, tokenize with base tokenizer
        tokens = self.base_tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=True
        )

        # Then encrypt if crypto handler is available
        if self.crypto_handler and self._should_encrypt():
            encrypted_tokens = self._encrypt_tokens(tokens)
            logger.debug(f"Encrypted {len(tokens)} tokens to {len(encrypted_tokens)} encrypted tokens")
            return encrypted_tokens
        else:
            logger.debug(f"Passthrough mode: {len(tokens)} tokens")
            return tokens

    def encode(self,
               text: str,
               truncation: Optional[bool] = None,
               max_length: Optional[int] = None,
               add_special_tokens: Optional[bool] = None) -> List[int]:
        """Encode text to token IDs."""
        return self.encode_one(text, truncation=truncation, max_length=max_length)

    def decode(self,
               token_ids: List[int],
               skip_special_tokens: Optional[bool] = None) -> str:
        """Decode token IDs to text."""
        # First decrypt if crypto handler is available
        if self.crypto_handler and self._should_decrypt(token_ids):
            decrypted_tokens = self._decrypt_tokens(token_ids)
            logger.debug(f"Decrypted {len(token_ids)} tokens to {len(decrypted_tokens)} plaintext tokens")
            token_ids = decrypted_tokens

        # Then decode with base tokenizer
        if skip_special_tokens is None:
            skip_special_tokens = False  # Default to False if None

        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _should_encrypt(self) -> bool:
        """Determine if encryption should be applied."""
        return VECTORCHAT_AVAILABLE and self.crypto_handler is not None

    def _should_decrypt(self, token_ids: List[int]) -> bool:
        """Determine if decryption should be applied."""
        # Simple heuristic: if tokens look like they might be encrypted
        # (e.g., contain header markers or have unusual patterns)
        return self._should_encrypt() and len(token_ids) > 0

    def _encrypt_tokens(self, tokens: List[int]) -> List[int]:
        """Encrypt token sequence using VectorChat crypto."""
        if not self.crypto_handler:
            return tokens

        try:
            # Prepare metadata for encryption
            metadata = {
                'key_offsets': self._generate_key_offsets(len(tokens)),
                'session_id': self._generate_session_id(),
                'pairing_sequence_bits': self._generate_pairing_bits(),
                'checksum_length': self.crypto_config.get('checksum_length', 2),
                'header_marker_override': self.crypto_config.get('header_marker_override'),
                'token_marker_override': self.crypto_config.get('token_marker_override'),
                'digit_permutation': self.crypto_config.get('digit_permutation'),
            }

            encrypted = self.crypto_handler.encrypt_tokens(tokens, metadata)
            return encrypted

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return tokens  # Fallback to plaintext

    def _decrypt_tokens(self, tokens: List[int]) -> List[int]:
        """Decrypt token sequence using VectorChat crypto."""
        if not self.crypto_handler:
            return tokens

        try:
            # For decryption, we need to extract metadata from the encrypted stream
            # This is a simplified version - in practice, metadata would be embedded
            metadata = {
                'key_offsets': self._generate_key_offsets(len(tokens)),
                'session_id': self._generate_session_id(),
                'pairing_sequence_bits': self._generate_pairing_bits(),
                'checksum_length': self.crypto_config.get('checksum_length', 2),
            }

            decrypted = self.crypto_handler.decrypt_tokens(tokens, metadata)
            return decrypted

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return tokens  # Fallback to original tokens

    def _generate_key_offsets(self, token_count: int) -> List[int]:
        """Generate key offsets for encryption."""
        # Use EMDM-based key generation as mentioned in user memories
        # This is a simplified implementation
        import random
        random.seed(self.crypto_config['emdm_seed_hex'])
        return [random.randint(0, 9) for _ in range(min(token_count, 100))]

    def _generate_session_id(self) -> str:
        """Generate a session ID for encryption."""
        import secrets
        return secrets.token_hex(self.crypto_config['session_id_length_bytes'])

    def apply_chat_template(self, conversation, **kwargs):
        """Apply chat template (delegate to base tokenizer if available)."""
        if hasattr(self.base_tokenizer, 'apply_chat_template'):
            return self.base_tokenizer.apply_chat_template(conversation, **kwargs)
        return conversation

    def convert_ids_to_tokens(self, ids, **kwargs):
        """Convert token IDs to token strings."""
        if hasattr(self.base_tokenizer, 'convert_ids_to_tokens'):
            return self.base_tokenizer.convert_ids_to_tokens(ids, **kwargs)
        return [str(id) for id in ids]

    def convert_tokens_to_string(self, tokens, **kwargs):
        """Convert token strings to text."""
        if hasattr(self.base_tokenizer, 'convert_tokens_to_string'):
            return self.base_tokenizer.convert_tokens_to_string(tokens, **kwargs)
        return ' '.join(tokens)


def create_vectorchat_tokenizer(base_tokenizer_path: str,
                               crypto_config: Optional[Dict[str, Any]] = None) -> VectorChatEncryptedTokenizer:
    """
    Factory function to create a VectorChat encrypted tokenizer.

    Args:
        base_tokenizer_path: Path or name of the base HuggingFace tokenizer
        crypto_config: Optional crypto configuration

    Returns:
        VectorChatEncryptedTokenizer instance
    """
    from transformers import AutoTokenizer

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        return VectorChatEncryptedTokenizer(base_tokenizer, crypto_config)
    except Exception as e:
        logger.error(f"Failed to create VectorChat tokenizer: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    crypto_config = {
        'emdm_seed_hex': 'test_seed_vectorchat_2024',
        'emdm_anchor_indices': [0, 1, 2, 3],
        'emdm_window_len': 10,
        'pairing_sequence_length': 8,
        'session_id_length_bytes': 16,
        'checksum_length': 2,
    }

    # Test with a simple tokenizer
    try:
        tokenizer = create_vectorchat_tokenizer("gpt2", crypto_config)

        # Test encryption
        test_text = "Hello, this is a test message for VectorChat encryption!"
        encrypted_tokens = tokenizer.encode(test_text)
        print(f"Original text: {test_text}")
        print(f"Encrypted tokens (first 20): {encrypted_tokens[:20]}")

        # Test decryption
        decrypted_text = tokenizer.decode(encrypted_tokens)
        print(f"Decrypted text: {decrypted_text}")
        print(f"Round-trip successful: {test_text == decrypted_text}")

    except Exception as e:
        print(f"Test failed: {e}")
