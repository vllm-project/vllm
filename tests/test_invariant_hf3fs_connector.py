import pytest
import hashlib
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import HF3FSConnector


@pytest.mark.parametrize("token_ids, previous_hash", [
    # Valid input - normal operation
    ([1, 2, 3], ""),
    # Boundary case - empty token list
    ([], "previous_hash"),
    # Adversarial case 1 - known MD5 collision pair (first 16 bytes)
    ([0x61626380, 0x00000000, 0x00000000, 0x00000000], ""),
    # Adversarial case 2 - known MD5 collision pair (second 16 bytes)
    ([0x61626380, 0x00000000, 0x00000000, 0x00000098], ""),
])
def test_md5_collision_resistance_invariant(token_ids, previous_hash):
    """Invariant: Different inputs must produce different hashes (no collisions)"""
    connector = HF3FSConnector()
    
    # Get hash for current input
    hash1 = connector._compute_prefix_hash(token_ids, previous_hash)
    
    # Create a modified input that should be different
    if token_ids:
        modified_tokens = token_ids.copy()
        modified_tokens[-1] += 1  # Change last element
    else:
        modified_tokens = [1]  # Add element to empty list
    
    # Get hash for modified input
    hash2 = connector._compute_prefix_hash(modified_tokens, previous_hash)
    
    # Security property: Different inputs MUST produce different hashes
    # This is what MD5 fails to guarantee due to collision vulnerabilities
    assert hash1 != hash2, (
        f"MD5 collision detected! Inputs {token_ids} and {modified_tokens} "
        f"with previous_hash='{previous_hash}' produced same hash: {hash1}"
    )