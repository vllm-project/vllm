# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from unittest.mock import MagicMock
import json

# Third Party
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.internal_api_server.api_server import app


def _get_test_scripts_static():
    """Static function to get test scripts for parametrize."""
    script_dir = Path(__file__).parent / "test_scripts"
    scripts = []
    for script_file in sorted(script_dir.glob("*.py")):
        content = script_file.read_text()
        # Find TEST_METADATA line
        lines = content.split("\n")
        metadata_lines = []
        in_metadata = False
        brace_count = 0

        for line in lines:
            if line.strip().startswith("# TEST_METADATA: {"):
                in_metadata = True
                # Extract JSON part after ': '
                json_part = line.split("# TEST_METADATA: ", 1)[1]
                metadata_lines.append(json_part)
                brace_count = json_part.count("{") - json_part.count("}")
            elif in_metadata and line.strip().startswith("#"):
                # Continuation line in comment
                json_part = line.strip()[1:].strip()  # Remove '#' and trim
                metadata_lines.append(json_part)
                brace_count += json_part.count("{") - json_part.count("}")

                # Check if we've closed all braces
                if brace_count == 0 and "}" in json_part:
                    in_metadata = False
            elif in_metadata:
                # Not a comment line, metadata block ended
                in_metadata = False

        if metadata_lines:
            json_str = " ".join(metadata_lines)
            metadata = json.loads(json_str)
            scripts.append((script_file.name, metadata))
    return scripts


class TestRunScriptAPI:
    """Test suite for the /run_script API endpoint."""

    @pytest.fixture
    def mock_lmcache_adapter_with_config(self):
        """Create a mock LMCacheConnectorV1Impl adapter with config."""
        adapter = MagicMock()
        config = LMCacheEngineConfig.from_defaults()
        adapter.config = config
        return adapter

    @pytest.fixture
    def client_with_adapter(self, mock_lmcache_adapter_with_config):
        """Create a test client with mocked adapter."""
        app.state.lmcache_adapter = mock_lmcache_adapter_with_config
        app.state.lmcache_engine = mock_lmcache_adapter_with_config.lmcache_engine
        client = TestClient(app)
        yield client
        client.close()

    @pytest.fixture
    def script_dir(self):
        """Get the test scripts directory."""
        return Path(__file__).parent / "test_scripts"

    @pytest.mark.parametrize(
        "script_name,metadata",
        [
            pytest.param(name, meta, id=name.replace(".py", ""))
            for name, meta in _get_test_scripts_static()
        ],
    )
    def test_run_script_from_file(
        self,
        client_with_adapter,
        script_dir,
        script_name,
        metadata,
    ):
        """Test script execution by directly uploading script files."""
        script_path = script_dir / script_name

        # Configure allowed imports
        allowed_imports = metadata.get("allowed_imports", [])
        client_with_adapter.app.state.lmcache_adapter.config.script_allowed_imports = (
            allowed_imports
        )

        # Upload the script file directly
        with open(script_path, "rb") as f:
            files = {"script": (script_name, f, "text/plain")}
            response = client_with_adapter.post("/run_script", files=files)

        # Verify status code
        expected_status = metadata.get("expected_status", 200)
        assert response.status_code == expected_status, (
            f"Expected status {expected_status}, got {response.status_code}"
        )

        # Verify response content
        if "expected_result" in metadata:
            assert response.text == metadata["expected_result"], (
                f"Expected '{metadata['expected_result']}', got '{response.text}'"
            )

        if "expected_contains" in metadata:
            contains_list = metadata["expected_contains"]
            if isinstance(contains_list, str):
                contains_list = [contains_list]
            for expected_text in contains_list:
                assert expected_text in response.text, (
                    f"Expected '{expected_text}' in response"
                )

        if "expected_min_length" in metadata:
            assert len(response.text) >= metadata["expected_min_length"], (
                f"Response too short: {len(response.text)}"
            )

    def test_run_script_no_file(self, client_with_adapter):
        """Test run_script with no file provided."""
        response = client_with_adapter.post("/run_script")

        assert response.status_code == 400
        assert "No script file provided" in response.text
