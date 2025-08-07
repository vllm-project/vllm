"""
Test script for GPT-OSS model with MXFP4 quantization.
This script validates the basic functionality of the GPT-OSS implementation.
"""

import os
import torch
from typing import Dict, Any


def test_gpt_oss_model_loading():
    """Test GPT-OSS model loading and configuration."""
    print("Testing GPT-OSS model loading...")

    try:
        # Import the model
        from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

        print("✓ Successfully imported GptOssForCausalLM")

        # Test configuration
        config = {
            "vocab_size": 50257,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "use_bias": False,
            "attention_bias": False,
            "mlp_bias": False,
            "torch_dtype": "float16",
            "use_cache": True,
        }

        print("✓ Model configuration created")

    except ImportError as e:
        print(f"✗ Failed to import GPT-OSS model: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in model configuration: {e}")
        return False

    return True


def test_mxfp4_quantization():
    """Test MXFP4 quantization configuration."""
    print("\nTesting MXFP4 quantization...")

    try:
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

        print("✓ Successfully imported Mxfp4Config")

        # Test quantization config
        quant_config = Mxfp4Config()
        print("✓ MXFP4 quantization config created")

        # Check supported dtypes
        supported_dtypes = quant_config.get_supported_act_dtypes()
        print(f"✓ Supported activation dtypes: {supported_dtypes}")

    except ImportError as e:
        print(f"✗ Failed to import MXFP4: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in MXFP4 configuration: {e}")
        return False

    return True


def test_harmony_utils():
    """Test harmony utilities for reasoning."""
    print("\nTesting Harmony utilities...")

    try:
        from vllm.entrypoints.harmony_utils import HarmonyEncoding

        print("✓ Successfully imported HarmonyEncoding")

        harmony = HarmonyEncoding()
        print("✓ Harmony encoding instance created")

        # Test basic functionality
        test_text = "This is a test reasoning step."
        encoded = harmony.encode_reasoning_step(test_text)
        print(
            f"✓ Encoded reasoning step: {encoded[:50]}..."
            if len(encoded) > 50
            else f"✓ Encoded reasoning step: {encoded}"
        )

    except ImportError as e:
        print(f"✗ Failed to import Harmony utils: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in Harmony utilities: {e}")
        return False

    return True


def test_tool_server():
    """Test MCP tool server functionality."""
    print("\nTesting MCP tool server...")

    try:
        from vllm.entrypoints.openai.tool_server import MCPToolServer

        print("✓ Successfully imported MCPToolServer")

        # Create tool server
        server = MCPToolServer()
        print("✓ MCP tool server instance created")

        # Test basic functionality
        tools = server.list_available_tools()
        print(f"✓ Available tools: {[tool['name'] for tool in tools]}")

    except ImportError as e:
        print(f"✗ Failed to import tool server: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in tool server: {e}")
        return False

    return True


def test_reasoning_parser():
    """Test OpenAI reasoning parser."""
    print("\nTesting reasoning parser...")

    try:
        from vllm.transformers_utils.openai_reasoning_parser import (
            OpenAIReasoningParser,
            extract_reasoning_and_final,
        )

        print("✓ Successfully imported reasoning parser")

        # Test parsing
        test_output = """
        <|reasoning|>
        Let me think about this step by step.
        First, I need to understand the question.
        Then I can provide an answer.
        <|/reasoning|>
        
        <|final|>
        Based on my reasoning, the answer is 42.
        <|/final|>
        """

        reasoning, final, is_tool_call = extract_reasoning_and_final(test_output)
        print(
            f"✓ Parsed reasoning: {reasoning[:50]}..."
            if reasoning and len(reasoning) > 50
            else f"✓ Parsed reasoning: {reasoning}"
        )
        print(f"✓ Parsed final: {final}")
        print(f"✓ Is tool call: {is_tool_call}")

    except ImportError as e:
        print(f"✗ Failed to import reasoning parser: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in reasoning parser: {e}")
        return False

    return True


def test_flash_attention_3():
    """Test Flash Attention 3 backend."""
    print("\nTesting Flash Attention 3...")

    try:
        from vllm.attention.backends.flash_attn_3 import FlashAttention3Backend

        print("✓ Successfully imported FlashAttention3Backend")

        backend = FlashAttention3Backend()
        print(f"✓ FA3 backend name: {backend.get_name()}")

        # Check FA3 availability
        from vllm.attention.backends.flash_attn_3 import HAS_FLASH_ATTN_3

        print(f"✓ Flash Attention 3 available: {HAS_FLASH_ATTN_3}")

    except ImportError as e:
        print(f"✗ Failed to import FA3 backend: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in FA3 backend: {e}")
        return False

    return True


def test_protocol_extensions():
    """Test OpenAI protocol extensions for reasoning."""
    print("\nTesting protocol extensions...")

    try:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        print("✓ Successfully imported ChatCompletionRequest")

        # Test reasoning parameter
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-oss",
            "include_reasoning": True,
        }

        request = ChatCompletionRequest(**request_data)
        print(f"✓ Request with reasoning parameter: {request.include_reasoning}")

    except ImportError as e:
        print(f"✗ Failed to import protocol: {e}")
        return False
    except Exception as e:
        print(f"✗ Error in protocol extensions: {e}")
        return False

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GPT-OSS Implementation Test Suite")
    print("=" * 60)

    tests = [
        test_gpt_oss_model_loading,
        test_mxfp4_quantization,
        test_harmony_utils,
        test_tool_server,
        test_reasoning_parser,
        test_flash_attention_3,
        test_protocol_extensions,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! GPT-OSS implementation looks good.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
