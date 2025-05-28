# SPDX-License-Identifier: Apache-2.0
import filecmp
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (  # noqa
    SharedStorageConnector)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_CONTEXT = "Hi " * 100
PROMPTS = [
    PROMPT_CONTEXT + "Hello, my name is",
    PROMPT_CONTEXT + "The capital of France is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=20)


class TestSharedStorageConnector(SharedStorageConnector):

    def __init__(self, config: VllmConfig, role):
        self.name = config.kv_transfer_config.kv_connector_extra_config["name"]
        self._connector = SharedStorageConnector(config, role)
        self.call_record: dict[str, int] = defaultdict(int)
        # Use a unique temp file per connector
        self._event_file = tempfile.gettempdir(
        ) + f"/connector_{self.name}_events.log"
        # Start with an empty file
        with open(self._event_file, "w") as _:
            pass

    def __getattribute__(self, name):
        if name in ("_connector", "call_record", "name", "_event_file",
                    "__class__", "__dict__", "__getattribute__",
                    "__init__"):  # avoid recursion
            return object.__getattribute__(self, name)
        if not hasattr(self._connector, name):
            return object.__getattribute__(self, name)
        attr = getattr(self._connector, name)

        # Intercept calls to the connector interface and write an event
        # for each one to a file, which can be read back in the main test proc.
        if callable(attr):

            def wrapper(*args, **kwargs):
                self.call_record[name] += 1
                # Log the event as a line to the file
                try:
                    with open(self._event_file, "a") as f:
                        f.write(name + "\n")
                except Exception as e:
                    print(f"[ERROR] Could not log event {name} "
                          f"for {self.name}: {e}")
                return attr(*args, **kwargs)

            return wrapper
        return attr


KVConnectorFactory.register_connector("TestSharedStorageConnector",
                                      TestSharedStorageConnector.__module__,
                                      TestSharedStorageConnector.__name__)


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
                "kv_connector": "TestSharedStorageConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(storage_1_path),
                    "name": "storage1",
                }
            }, {
                "kv_connector": "TestSharedStorageConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(storage_2_path),
                    "name": "storage2",
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

    events = get_connector_events()
    # get_num_new_matched_tokens will be called on each connector in turn.
    # neither of them have hits so update_state_after_alloc won't be called.
    assert events["storage1"][:3] == [
        'get_num_new_matched_tokens', 'build_connector_meta',
        'bind_connector_metadata'
    ]
    assert events["storage2"][:3] == [
        'get_num_new_matched_tokens', 'build_connector_meta',
        'bind_connector_metadata'
    ]

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will return new tokens from the first
    # connector so update_state_after_alloc will be called once blocks
    # are allocated for the first connector.
    # get_num_new_matched_tokens *won't* be called on the second connector
    # in this case.
    assert events["storage1"][:4] == [
        'get_num_new_matched_tokens', 'update_state_after_alloc',
        'build_connector_meta', 'bind_connector_metadata'
    ]
    assert events["storage2"][:2] == [
        'build_connector_meta', 'bind_connector_metadata'
    ]

    # Delete storage1 connector state
    shutil.rmtree(storage_1_path)

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will be called for the first connector but it
    # won't have a hit so update_state_after_alloc won't be called.
    # get_num_new_matched_tokens will also be called on the second connector,
    # but it should have a hit so update_state_after_alloc will be called.
    assert events["storage1"][:3] == [
        'get_num_new_matched_tokens', 'build_connector_meta',
        'bind_connector_metadata'
    ]
    assert events["storage2"][:4] == [
        'get_num_new_matched_tokens', 'update_state_after_alloc',
        'build_connector_meta', 'bind_connector_metadata'
    ]

    # Clean up
    shutil.rmtree(storage_1_path)
    shutil.rmtree(storage_2_path)


def get_connector_events() -> dict[str, list[str]]:
    # Read in connector events and reset the files.
    import glob
    event_files = glob.glob(tempfile.gettempdir() + "/connector_*_events.log")
    connector_events = {}
    for fname in event_files:
        name = fname.split("connector_")[1].split("_events.log")[0]
        try:
            with open(fname, "r+") as f:
                connector_events[name] = [
                    line.strip() for line in f if line.strip()
                ]
                f.truncate(0)
        except Exception as e:
            print(f"[ERROR] Could not read connector events for {name}: {e}")

    return connector_events


def test_engine_id_conflict():
    configs = [KVTransferConfig() for _ in range(2)]
    ids = [config.engine_id for config in configs]
    assert ids[0] != ids[1], (
        "Engine IDs should be different for different configs. "
        f"Got {ids}")
