# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Functional tests for Helix parallelism and standard DCP.

This test suite validates correctness of all four operational modes:
1. Standard DCP + GQA models
2. Standard DCP + MLA models  
3. Helix + GQA models
4. Helix + MLA models

Requirements:
- 4+ GPUs with compute capability 9.0+ (Hopper/Blackwell)
- ~20GB VRAM per GPU for the test models

Run instructions:
    # Run all functional tests
    pytest tests/distributed/test_helix_functional.py -v -s
    
    # Run specific test class
    pytest tests/distributed/test_helix_functional.py::TestHelixGQA -v -s
    pytest tests/distributed/test_helix_functional.py::TestHelixMLA -v -s
    pytest tests/distributed/test_helix_functional.py::TestStandardDCP -v -s
    
    # Run with specific number of GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 pytest tests/distributed/test_helix_functional.py -v -s
"""

import os
from typing import NamedTuple

import pytest
import torch

from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
from vllm.logger import init_logger

logger = init_logger("test_helix_functional")

# =============================================================================
# Test Configuration
# =============================================================================

# Test models
# - GQA model: Small enough for testing, has multiple KV heads
# - MLA model: Uses Multi-head Latent Attention (DeepSeek architecture)
GQA_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # num_kv_heads=2
MLA_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"  # MLA architecture

# Sanity check prompts - designed to detect gibberish
SANITY_PROMPTS = [
    # Simple math (deterministic answer)
    "What is 2 + 2? Answer with just the number:",
    # Logical reasoning
    "If all cats are animals, and Fluffy is a cat, then Fluffy is a(n):",
    # Common knowledge
    "The capital of France is:",
]

# Expected patterns in responses (should NOT be gibberish)
EXPECTED_PATTERNS = [
    ["4", "four"],
    ["animal"],
    ["Paris", "paris"],
]


class TestConfig(NamedTuple):
    """Configuration for a single test case."""
    tp_size: int
    dcp_size: int
    helix_mode: bool
    attn_backend: str
    model_type: str  # "gqa" or "mla"
    description: str


# =============================================================================
# Test Configurations Matrix
# =============================================================================

# GQA Model Configurations (Qwen2.5-1.5B: 12 Q heads, 2 KV heads)
GQA_CONFIGS = [
    # Standard DCP (helix_mode=False)
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=False,
        attn_backend="FLASH_ATTN",
        model_type="gqa",
        description="GQA + Standard DCP + FlashAttn",
    ),
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=False,
        attn_backend="FLASHINFER",
        model_type="gqa",
        description="GQA + Standard DCP + FlashInfer",
    ),
    # Helix mode (helix_mode=True)
    # TP=4, DCP=2 -> TPA=2, KVP=2 (valid for Qwen with 2 KV heads)
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=True,
        attn_backend="FLASH_ATTN",
        model_type="gqa",
        description="GQA + Helix (TPA=2, KVP=2) + FlashAttn",
    ),
    TestConfig(
        tp_size=4,
        dcp_size=2,
        helix_mode=True,
        attn_backend="FLASHINFER",
        model_type="gqa",
        description="GQA + Helix (TPA=2, KVP=2) + FlashInfer",
    ),
]

# MLA Model Configurations (DeepSeek-V2-Lite: MLA attention)
MLA_CONFIGS = [
    # Standard DCP (helix_mode=False)
    TestConfig(
        tp_size=4,
        dcp_size=4,
        helix_mode=False,
        attn_backend="FLASHMLA",
        model_type="mla",
        description="MLA + Standard DCP + FlashMLA",
    ),
    # Helix mode (helix_mode=True)
    # TP=4, DCP=4 -> TPA=1, KVP=4 (valid for MLA)
    TestConfig(
        tp_size=4,
        dcp_size=4,
        helix_mode=True,
        attn_backend="FLASHMLA",
        model_type="mla",
        description="MLA + Helix (TPA=1, KVP=4) + FlashMLA",
    ),
]


# =============================================================================
# Helper Functions
# =============================================================================

def check_gpu_requirements(tp_size: int, min_compute_capability: tuple = (9, 0)):
    """Check if GPU requirements are met, skip test if not."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs, have {num_gpus}")
    
    cc = torch.cuda.get_device_capability()
    if cc < min_compute_capability:
        pytest.skip(
            f"Need compute capability {min_compute_capability}, have {cc}"
        )


def validate_response(response_text: str, expected_patterns: list[str]) -> bool:
    """Check if response contains any of the expected patterns."""
    response_lower = response_text.lower()
    return any(pattern.lower() in response_lower for pattern in expected_patterns)


def is_gibberish(text: str) -> bool:
    """Heuristic check for gibberish output."""
    if not text or len(text.strip()) == 0:
        return True
    
    # Check for excessive special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
    if special_ratio > 0.5:
        return True
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:  # >80% repetition
            return True
    
    return False


def run_sanity_check(
    client,
    model_id: str,
    config: TestConfig,
) -> dict:
    """Run sanity check prompts and validate responses."""
    results = {
        "config": config.description,
        "passed": True,
        "responses": [],
        "errors": [],
    }
    
    for i, (prompt, expected) in enumerate(zip(SANITY_PROMPTS, EXPECTED_PATTERNS)):
        try:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50,
                temperature=0,  # Deterministic
            )
            text = response.choices[0].text.strip()
            results["responses"].append(text)
            
            # Check for gibberish
            if is_gibberish(text):
                results["passed"] = False
                results["errors"].append(f"Prompt {i}: Gibberish detected: '{text}'")
                continue
            
            # Check for expected pattern
            if not validate_response(text, expected):
                # Not a failure, just a warning (model might phrase differently)
                logger.warning(
                    f"Prompt {i}: Expected one of {expected}, got '{text}'"
                )
        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Prompt {i}: Exception: {e}")
    
    return results


def build_server_args(
    config: TestConfig,
    model_id: str,
    trust_remote_code: bool = False,
) -> list[str]:
    """Build vLLM server arguments for a test configuration."""
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "32",
        "--tensor-parallel-size", str(config.tp_size),
        "--decode-context-parallel-size", str(config.dcp_size),
        "--attention-backend", config.attn_backend,
        "--enable-chunked-prefill",
    ]
    
    if config.helix_mode:
        args.append("--helix-mode")
    
    if trust_remote_code:
        args.append("--trust-remote-code")
    
    # KV cache interleave size for DCP
    if config.model_type == "mla":
        args.extend(["--cp-kv-cache-interleave-size", "64"])
    else:
        args.extend(["--cp-kv-cache-interleave-size", "16"])
    
    return args


# =============================================================================
# Test Classes
# =============================================================================

class TestHelixGQA:
    """Test Helix mode with GQA models (e.g., Qwen, Llama)."""
    
    @pytest.mark.parametrize("config", [c for c in GQA_CONFIGS if c.helix_mode])
    @create_new_process_for_each_test()
    def test_helix_gqa_sanity(self, config: TestConfig, num_gpus_available):
        """Test Helix GQA mode produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        
        server_args = build_server_args(config, GQA_MODEL, trust_remote_code=False)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, GQA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestHelixMLA:
    """Test Helix mode with MLA models (e.g., DeepSeek-V2)."""
    
    @pytest.mark.parametrize("config", [c for c in MLA_CONFIGS if c.helix_mode])
    @create_new_process_for_each_test()
    def test_helix_mla_sanity(self, config: TestConfig, num_gpus_available):
        """Test Helix MLA mode produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        
        server_args = build_server_args(config, MLA_MODEL, trust_remote_code=True)
        
        with RemoteOpenAIServer(MLA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, MLA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestStandardDCP:
    """Test standard DCP mode (helix_mode=False) - baseline."""
    
    @pytest.mark.parametrize("config", [c for c in GQA_CONFIGS if not c.helix_mode])
    @create_new_process_for_each_test()
    def test_standard_dcp_gqa_sanity(self, config: TestConfig, num_gpus_available):
        """Test standard DCP with GQA model produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        
        server_args = build_server_args(config, GQA_MODEL, trust_remote_code=False)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, GQA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"
    
    @pytest.mark.parametrize("config", [c for c in MLA_CONFIGS if not c.helix_mode])
    @create_new_process_for_each_test()
    def test_standard_dcp_mla_sanity(self, config: TestConfig, num_gpus_available):
        """Test standard DCP with MLA model produces coherent output."""
        check_gpu_requirements(config.tp_size)
        
        logger.info(f"Testing: {config.description}")
        
        server_args = build_server_args(config, MLA_MODEL, trust_remote_code=True)
        
        with RemoteOpenAIServer(MLA_MODEL, server_args, num_gpus=config.tp_size) as server:
            client = server.get_client()
            results = run_sanity_check(client, MLA_MODEL, config)
        
        logger.info(f"Results: {results}")
        
        assert results["passed"], f"Sanity check failed: {results['errors']}"


class TestHelixVsDCPConsistency:
    """Test that Helix produces outputs consistent with standard DCP."""
    
    @create_new_process_for_each_test()
    def test_gqa_helix_vs_dcp_consistency(self, num_gpus_available):
        """Verify Helix GQA output matches standard DCP output."""
        check_gpu_requirements(4)
        
        prompts = ["What is the capital of Japan?"]
        
        # Standard DCP
        dcp_config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=False,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="baseline",
        )
        dcp_args = build_server_args(dcp_config, GQA_MODEL)
        
        with RemoteOpenAIServer(GQA_MODEL, dcp_args, num_gpus=4) as server:
            client = server.get_client()
            dcp_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=GQA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                dcp_outputs.append(response.choices[0].text)
        
        # Helix
        helix_config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=True,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="helix",
        )
        helix_args = build_server_args(helix_config, GQA_MODEL)
        
        with RemoteOpenAIServer(GQA_MODEL, helix_args, num_gpus=4) as server:
            client = server.get_client()
            helix_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=GQA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                helix_outputs.append(response.choices[0].text)
        
        # Compare
        for i, (dcp_out, helix_out) in enumerate(zip(dcp_outputs, helix_outputs)):
            logger.info(f"Prompt {i}:")
            logger.info(f"  DCP:   '{dcp_out}'")
            logger.info(f"  Helix: '{helix_out}'")
            
            # For exact match (both use exact attention, should be identical)
            assert dcp_out == helix_out, (
                f"Output mismatch for prompt {i}:\n"
                f"DCP: {dcp_out}\n"
                f"Helix: {helix_out}"
            )
    
    @create_new_process_for_each_test()
    def test_mla_helix_vs_dcp_consistency(self, num_gpus_available):
        """Verify Helix MLA output matches standard DCP output."""
        check_gpu_requirements(4)
        
        prompts = ["What is 5 + 5?"]
        
        # Standard DCP
        dcp_config = TestConfig(
            tp_size=4, dcp_size=4, helix_mode=False,
            attn_backend="FLASHMLA", model_type="mla",
            description="baseline",
        )
        dcp_args = build_server_args(dcp_config, MLA_MODEL, trust_remote_code=True)
        
        with RemoteOpenAIServer(MLA_MODEL, dcp_args, num_gpus=4) as server:
            client = server.get_client()
            dcp_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=MLA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                dcp_outputs.append(response.choices[0].text)
        
        # Helix
        helix_config = TestConfig(
            tp_size=4, dcp_size=4, helix_mode=True,
            attn_backend="FLASHMLA", model_type="mla",
            description="helix",
        )
        helix_args = build_server_args(helix_config, MLA_MODEL, trust_remote_code=True)
        
        with RemoteOpenAIServer(MLA_MODEL, helix_args, num_gpus=4) as server:
            client = server.get_client()
            helix_outputs = []
            for prompt in prompts:
                response = client.completions.create(
                    model=MLA_MODEL,
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0,
                )
                helix_outputs.append(response.choices[0].text)
        
        # Compare
        for i, (dcp_out, helix_out) in enumerate(zip(dcp_outputs, helix_outputs)):
            logger.info(f"Prompt {i}:")
            logger.info(f"  DCP:   '{dcp_out}'")
            logger.info(f"  Helix: '{helix_out}'")
            
            assert dcp_out == helix_out, (
                f"Output mismatch for prompt {i}:\n"
                f"DCP: {dcp_out}\n"
                f"Helix: {helix_out}"
            )


# =============================================================================
# Quick Smoke Test
# =============================================================================

class TestQuickSmoke:
    """Quick smoke test to verify basic functionality."""
    
    @create_new_process_for_each_test()
    def test_helix_gqa_quick(self, num_gpus_available):
        """Quick smoke test for Helix GQA."""
        check_gpu_requirements(4)
        
        config = TestConfig(
            tp_size=4, dcp_size=2, helix_mode=True,
            attn_backend="FLASH_ATTN", model_type="gqa",
            description="Helix GQA smoke test",
        )
        
        server_args = build_server_args(config, GQA_MODEL)
        
        with RemoteOpenAIServer(GQA_MODEL, server_args, num_gpus=4) as server:
            client = server.get_client()
            response = client.completions.create(
                model=GQA_MODEL,
                prompt="Hello, my name is",
                max_tokens=20,
                temperature=0,
            )
            text = response.choices[0].text
            logger.info(f"Response: '{text}'")
            assert not is_gibberish(text), f"Gibberish output: {text}"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
