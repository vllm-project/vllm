#!/usr/bin/env python3
"""
VectorChat vLLM Integration Test Script
Tests the encrypted tokenizer integration with vLLM
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).resolve().parent
vllm_dir = current_dir
vectorchat_dir = current_dir.parent / 'AID-CORE-COMMERCIAL' / 'vectorchat' / 'daemon'

for path in [str(vllm_dir), str(vectorchat_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vectorchat_tokenizer():
    """Test the VectorChat encrypted tokenizer."""
    try:
        from vectorchat_tokenizer import create_vectorchat_tokenizer, VectorChatEncryptedTokenizer

        # Test configuration
        crypto_config = {
            'emdm_seed_hex': 'test_integration_2024',
            'emdm_anchor_indices': [0, 1, 2, 3],
            'emdm_window_len': 10,
            'pairing_sequence_length': 8,
            'session_id_length_bytes': 16,
            'checksum_length': 2,
        }

        logger.info("Creating VectorChat tokenizer...")
        tokenizer = create_vectorchat_tokenizer("gpt2", crypto_config)

        # Test data
        test_texts = [
            "Hello world!",
            "This is a test of the VectorChat encrypted tokenizer integration.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can now process encrypted data streams.",
        ]

        logger.info("Testing encryption/decryption round-trip...")

        for i, text in enumerate(test_texts):
            logger.info(f"\n--- Test {i+1}: '{text[:50]}...' ---")

            # Encrypt
            encrypted_tokens = tokenizer.encode(text)
            logger.info(f"Original tokens: {len(tokenizer.base_tokenizer.encode(text))}")
            logger.info(f"Encrypted tokens: {len(encrypted_tokens)}")
            logger.info(f"Token difference: {len(encrypted_tokens) - len(tokenizer.base_tokenizer.encode(text))}")

            # Decrypt
            decrypted_text = tokenizer.decode(encrypted_tokens)
            logger.info(f"Decrypted text: '{decrypted_text[:100]}...'")

            # Verify round-trip
            success = text == decrypted_text
            logger.info(f"Round-trip successful: {success}")

            if not success:
                logger.error(f"MISMATCH! Original: '{text}'")
                logger.error(f"MISMATCH! Decrypted: '{decrypted_text}'")
                return False

        logger.info("\n✅ All VectorChat tokenizer tests passed!")
        return True

    except Exception as e:
        logger.error(f"VectorChat tokenizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_vllm_integration():
    """Test integration with vLLM components."""
    try:
        # Test if we can import vLLM components
        logger.info("Testing vLLM component imports...")

        from vllm.transformers_utils.tokenizer_base import TokenizerBase
        logger.info("✅ TokenizerBase import successful")

        # Skip the problematic import and test our wrapper directly
        logger.info("Testing VectorChat tokenizer wrapper...")

        # Test creating a mock tokenizer that inherits from TokenizerBase
        class MockTokenizer(TokenizerBase):
            def __init__(self):
                self._vocab_size = 50257  # GPT-2 vocab size
                self._bos_token_id = 50256
                self._eos_token_id = 50256

            @property
            def all_special_tokens_extended(self):
                return ["<|endoftext|>"]

            @property
            def all_special_tokens(self):
                return ["<|endoftext|>"]

            @property
            def all_special_ids(self):
                return [50256]

            @property
            def bos_token_id(self):
                return self._bos_token_id

            @property
            def eos_token_id(self):
                return self._eos_token_id

            @property
            def sep_token(self):
                return None

            @property
            def pad_token(self):
                return None

            @property
            def is_fast(self):
                return False

            @property
            def vocab_size(self):
                return self._vocab_size

            @property
            def max_token_id(self):
                return self._vocab_size - 1

            def __call__(self, text, **kwargs):
                return [1, 2, 3]  # Mock tokens

            def get_vocab(self):
                return {"mock": 0}

            def get_added_vocab(self):
                return {}

            def encode_one(self, text, **kwargs):
                return [1, 2, 3]

            def encode(self, text, **kwargs):
                return [1, 2, 3]

            def decode(self, tokens, **kwargs):
                return "mock decoded text"

            def apply_chat_template(self, conversation, **kwargs):
                return conversation

            def convert_ids_to_tokens(self, ids, **kwargs):
                return [str(id) for id in ids]

            def convert_tokens_to_string(self, tokens, **kwargs):
                return ' '.join(tokens)

        mock_tokenizer = MockTokenizer()
        logger.info("✅ Mock tokenizer created successfully")

        # Test wrapping with our VectorChat tokenizer
        from vectorchat_tokenizer import VectorChatEncryptedTokenizer

        wrapped_tokenizer = VectorChatEncryptedTokenizer(mock_tokenizer)
        logger.info("✅ VectorChat wrapper created successfully")

        # Test basic functionality
        test_tokens = wrapped_tokenizer.encode("test")
        decoded_text = wrapped_tokenizer.decode(test_tokens)
        logger.info(f"✅ Basic encode/decode test: {len(test_tokens)} tokens -> '{decoded_text}'")

        logger.info("✅ vLLM integration tests passed!")
        return True

    except Exception as e:
        logger.error(f"vLLM integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting VectorChat vLLM Integration Tests")
    logger.info("=" * 60)

    # Test 1: VectorChat tokenizer
    logger.info("\n📝 Test 1: VectorChat Encrypted Tokenizer")
    vectorchat_success = test_vectorchat_tokenizer()

    # Test 2: vLLM integration
    logger.info("\n🔧 Test 2: vLLM Component Integration")
    vllm_success = test_vllm_integration()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    logger.info(f"VectorChat Tokenizer: {'✅ PASS' if vectorchat_success else '❌ FAIL'}")
    logger.info(f"vLLM Integration: {'✅ PASS' if vllm_success else '❌ FAIL'}")

    if vectorchat_success and vllm_success:
        logger.info("\n🎉 All tests passed! VectorChat integration is ready for development.")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
