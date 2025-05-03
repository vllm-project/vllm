# SPDX-License-Identifier: Apache-2.0
import filecmp
import shutil
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_CONTEXT = "Hi " * 100
PROMPTS = [
    PROMPT_CONTEXT + "Hello, my name is",
    PROMPT_CONTEXT + "The capital of France is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=20)


# Helper function to compare directories recursively
def _compare_directories(dir1: Path, dir2: Path) -> bool:
    """Compares two directories recursively for identical content."""
    dcmp = filecmp.dircmp(dir1, dir2)
    if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
        print(f"Differences found between {dir1} and {dir2}:")
        print(f"  Left only: {dcmp.left_only}")
        print(f"  Right only: {dcmp.right_only}")
        print(f"  Different files: {dcmp.diff_files}")
        return False
    for sub_dir in dcmp.common_dirs:
        if not _compare_directories(dir1 / sub_dir, dir2 / sub_dir):
            return False
    return True


def test_multi_shared_storage_connector_consistency():
    """
    Tests that MultiConnector with two SharedStorageConnectors saves
    identical KV cache data to separate storage locations.
    """
    storage_1_path = Path("storage_1/")
    storage_2_path = Path("storage_2/")
    shutil.rmtree(storage_1_path, ignore_errors=True)
    shutil.rmtree(storage_2_path, ignore_errors=True)
    storage_1_path.mkdir()
    storage_2_path.mkdir()

    # Configure MultiConnector with two SharedStorageConnectors
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [{
                "kv_connector": "SharedStorageConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(storage_1_path)
                }
            }, {
                "kv_connector": "SharedStorageConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(storage_2_path)
                }
            }]
        },
    )

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
    )
    # Run generation - this should trigger saving KV cache
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    # --- Verification ---

    # Check that both storage directories were populated
    local_subdirs = list(storage_1_path.iterdir())
    external_subdirs = list(storage_2_path.iterdir())

    assert len(
        local_subdirs
    ) > 0, f"Local storage path {storage_1_path} is empty after generation."
    assert len(external_subdirs) > 0, (
        f"External storage path {storage_2_path} is empty after generation.")
    assert len(local_subdirs) == len(external_subdirs), (
        f"Mismatch in number of cache entries: "
        f"Local={len(local_subdirs)}, External={len(external_subdirs)}")

    # The subdirectories should correspond to the prompt hashes
    # Since prompts are the same, the hash directories should be the same name
    local_subdir_names = sorted([d.name for d in local_subdirs])
    external_subdir_names = sorted([d.name for d in external_subdirs])
    assert local_subdir_names == external_subdir_names, (
        "Cache directory names do not match between local and external storage"
    )

    # Compare the contents of each corresponding cache directory
    for subdir_name in local_subdir_names:
        print(f"Comparing contents of cache directory: {subdir_name}")
        assert _compare_directories(storage_1_path / subdir_name,
                                    storage_2_path / subdir_name), \
            (f"Contents differ for cache directory '{subdir_name}' between "
             f"{storage_1_path} and {storage_2_path}")

    # Clean up
    shutil.rmtree(storage_1_path)
    shutil.rmtree(storage_2_path)
