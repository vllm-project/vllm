# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RuntimePluginLauncher.
"""

# Standard
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

# Third Party
import pytest

# First Party
from lmcache.v1.plugin.runtime_plugin_launcher import RuntimePluginLauncher


class MockConfig:
    """Mock configuration object."""

    def __init__(self, runtime_plugin_locations=None):
        self.runtime_plugin_locations = runtime_plugin_locations or []

    def to_json(self):
        return '{"test": "config"}'


@pytest.fixture
def temp_plugin_dir():
    """Create a temporary directory for plugin files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def config_with_location(temp_plugin_dir):
    """Create a config with plugin location."""
    return MockConfig([temp_plugin_dir])


@pytest.fixture
def config_without_location():
    """Create a config without plugin location."""
    return MockConfig([])


def test_init():
    """Test RuntimePluginLauncher initialization."""
    config = MockConfig()
    role = "WORKER"
    worker_count = 4
    worker_id = 1

    launcher = RuntimePluginLauncher(config, role, worker_count, worker_id)

    assert launcher.config == config
    assert launcher.role == "WORKER"
    assert launcher.worker_count == worker_count
    assert launcher.worker_id == worker_id
    assert launcher.plugin_processes == []


def test_launch_plugins_no_locations(config_without_location):
    """Test launch_plugins when no locations are configured."""
    launcher = RuntimePluginLauncher(
        config_without_location, "WORKER", worker_count=4, worker_id=1
    )

    # Should not raise any exceptions
    launcher.launch_plugins()
    assert len(launcher.plugin_processes) == 0


def test_launch_plugins_location_does_not_exist():
    """Test launch_plugins when location does not exist."""
    config = MockConfig(["/non/existent/path"])
    launcher = RuntimePluginLauncher(config, "WORKER", worker_count=4, worker_id=1)

    with patch("lmcache.v1.plugin.runtime_plugin_launcher.logger") as mock_logger:
        launcher.launch_plugins()

        # Should log warning
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "does not exist" in call_args

    assert len(launcher.plugin_processes) == 0


@patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which")
@patch("lmcache.v1.plugin.runtime_plugin_launcher.subprocess.Popen")
def test_launch_plugins_python_file(
    mock_popen, mock_which, temp_plugin_dir, config_with_location
):
    """Test launching a Python plugin file."""
    # Create a Python plugin file
    plugin_file = Path(temp_plugin_dir) / "worker_test.py"
    plugin_content = "#!/usr/bin/env python3\nprint('Hello from plugin')"
    plugin_file.write_text(plugin_content)

    # Mock subprocess
    mock_proc = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stdout.readline = Mock(side_effect=["output line\n", ""])
    mock_proc.poll = Mock(return_value=None)
    mock_proc.returncode = 0
    mock_popen.return_value = mock_proc

    # Mock interpreter resolution
    mock_which.return_value = "/usr/bin/python3"

    launcher = RuntimePluginLauncher(
        config_with_location, "WORKER", worker_count=4, worker_id=1
    )

    with patch("threading.Thread") as mock_thread:
        launcher.launch_plugins()

        # Should create process
        assert mock_popen.called
        call_args = mock_popen.call_args[0][0]
        assert call_args[0] == "/usr/bin/python3"
        assert str(call_args[1]) == str(plugin_file)

        # Should start thread for output capture
        assert mock_thread.called

        # Should add process to list
        assert len(launcher.plugin_processes) == 1
        assert launcher.plugin_processes[0] == mock_proc


@patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which")
@patch("lmcache.v1.plugin.runtime_plugin_launcher.subprocess.Popen")
def test_launch_plugins_role_filtering(
    mock_popen,
    mock_which,
    temp_plugin_dir,
    config_with_location,
):
    """Test plugin filtering based on role."""
    # Create plugins for different roles
    worker_plugin = Path(temp_plugin_dir) / "worker_test.py"
    scheduler_plugin = Path(temp_plugin_dir) / "scheduler_test.py"
    all_plugin = Path(temp_plugin_dir) / "all_test.py"

    for plugin in [worker_plugin, scheduler_plugin, all_plugin]:
        plugin.write_text("#!/usr/bin/env python3\n")

    # Mock subprocess
    mock_proc = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stdout.readline = Mock(return_value="")
    mock_proc.poll = Mock(return_value=None)
    mock_popen.return_value = mock_proc

    # Mock interpreter resolution
    mock_which.return_value = "/usr/bin/python3"

    # Test with worker role - should launch worker_plugin and all_plugin
    launcher = RuntimePluginLauncher(
        config_with_location, "WORKER", worker_count=4, worker_id=1
    )

    with (
        patch("lmcache.v1.plugin.runtime_plugin_launcher.logger") as mock_logger,
        patch("threading.Thread"),
    ):
        launcher.launch_plugins()

        # Should log info about skipping scheduler plugin
        # New format: logger.info("Skipping %s: requires role %s", file, role)
        info_calls = [call[0] for call in mock_logger.info.call_args_list]
        scheduler_skipped = any(
            len(call) >= 3
            and "requires role" in call[0]
            and "SCHEDULER" in str(call[2])
            for call in info_calls
        )
        assert scheduler_skipped

        # Should launch 2 plugins (worker_plugin and all_plugin)
        assert mock_popen.call_count == 2


@patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which")
@patch("lmcache.v1.plugin.runtime_plugin_launcher.subprocess.Popen")
def test_launch_plugins_worker_id_filtering(
    mock_popen, mock_which, temp_plugin_dir, config_with_location
):
    """Test plugin filtering based on worker ID."""
    # Create plugins for specific worker IDs
    worker0_plugin = Path(temp_plugin_dir) / "worker_0_specific.py"
    worker1_plugin = Path(temp_plugin_dir) / "worker_1_specific.py"
    generic_worker_plugin = Path(temp_plugin_dir) / "worker_generic.py"

    for plugin in [worker0_plugin, worker1_plugin, generic_worker_plugin]:
        plugin.write_text("#!/usr/bin/env python3\n")

    # Mock subprocess
    mock_proc = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stdout.readline = Mock(return_value="")
    mock_proc.poll = Mock(return_value=None)
    mock_popen.return_value = mock_proc

    # Mock interpreter resolution
    mock_which.return_value = "/usr/bin/python3"

    # Test with worker_id=1 - should launch worker1_plugin and generic_worker_plugin
    launcher = RuntimePluginLauncher(
        config_with_location, "WORKER", worker_count=4, worker_id=1
    )

    with (
        patch("lmcache.v1.plugin.runtime_plugin_launcher.logger") as mock_logger,
        patch("threading.Thread"),
    ):
        launcher.launch_plugins()

        # Should log info about skipping worker0 plugin
        info_calls = [call[0] for call in mock_logger.info.call_args_list]
        worker0_skipped = any(
            len(call) >= 4
            and "is skipping" in call[0]
            and call[1] == 1
            and call[3] == 0
            for call in info_calls
        )
        assert worker0_skipped

        # Should launch 2 plugins (worker1_plugin and generic_worker_plugin)
        assert mock_popen.call_count == 2


def test_get_interpreter_python(temp_plugin_dir):
    """Test interpreter detection for Python files."""
    # Create a Python file with shebang
    python_file = Path(temp_plugin_dir) / "test.py"
    python_file.write_text("#!/usr/bin/env python3\nprint('test')")

    config = MockConfig()
    launcher = RuntimePluginLauncher(config, "WORKER", 4, 1)

    with patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which") as mock_which:
        # Mock which to return the interpreter for shebang
        mock_which.return_value = "/usr/bin/python3"
        interpreter = launcher._get_interpreter(python_file)

        assert interpreter == "/usr/bin/python3"
        # Should try shebang interpreter first
        mock_which.assert_called_once_with("/usr/bin/env python3")
        # Should NOT try fallback interpreters since shebang interpreter is found


def test_get_interpreter_bash(temp_plugin_dir):
    """Test interpreter detection for Bash files."""
    # Create a Bash file
    bash_file = Path(temp_plugin_dir) / "test.sh"
    bash_file.write_text("#!/bin/bash\necho 'test'")

    config = MockConfig()
    launcher = RuntimePluginLauncher(config, "WORKER", 4, 1)

    with patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which") as mock_which:
        # Mock which to return the interpreter for shebang
        mock_which.return_value = "/bin/bash"
        interpreter = launcher._get_interpreter(bash_file)

        assert interpreter == "/bin/bash"
        # Should try shebang interpreter first
        mock_which.assert_called_once_with("/bin/bash")
        # Should NOT try fallback bash since shebang interpreter is found


def test_get_interpreter_no_shebang(temp_plugin_dir):
    """Test interpreter detection for file without shebang."""
    # Create a Python file without shebang
    python_file = Path(temp_plugin_dir) / "test.py"
    python_file.write_text("print('test')")

    config = MockConfig()
    launcher = RuntimePluginLauncher(config, "WORKER", 4, 1)

    with patch("lmcache.v1.plugin.runtime_plugin_launcher.shutil.which") as mock_which:
        # Mock which to return interpreter for python
        mock_which.side_effect = {
            "python": "/usr/bin/python",
            "python3": "/usr/bin/python3",
        }.get
        interpreter = launcher._get_interpreter(python_file)

        # Should return python (first in the fallback list)
        assert interpreter == "/usr/bin/python"
        # Should try python first, then stop since it's found
        mock_which.assert_called_once_with("python")


def test_get_interpreter_unsupported_type(temp_plugin_dir):
    """Test interpreter detection for unsupported file type."""
    # Create an unsupported file type
    unsupported_file = Path(temp_plugin_dir) / "test.txt"
    unsupported_file.write_text("test content")

    config = MockConfig()
    launcher = RuntimePluginLauncher(config, "WORKER", 4, 1)

    with pytest.raises(ValueError) as exc_info:
        launcher._get_interpreter(unsupported_file)

    assert "not supported" in str(exc_info.value)


@pytest.mark.parametrize(
    ("role_name", "worker_id", "filename_parts", "should_skip"),
    [
        ("WORKER", 1, ["SCHEDULER", "test"], True),
        ("WORKER", 1, ["WORKER", "test"], False),
        ("WORKER", 1, ["ALL", "test"], False),
        ("WORKER", 1, ["WORKER", "0", "specific"], True),
        ("WORKER", 1, ["WORKER", "1", "specific"], False),
        ("WORKER", 1, ["WORKER", "generic"], False),
        ("SCHEDULER", 0, ["WORKER", "generic"], True),
    ],
)
def test_should_skip_plugin(role_name, worker_id, filename_parts, should_skip):
    """Test plugin skipping logic."""
    config = MockConfig()
    launcher = RuntimePluginLauncher(config, role_name, 4, worker_id)

    assert launcher._should_skip_plugin(Path("test.py"), filename_parts) is should_skip


def test_stop_plugins():
    """Test stopping plugin processes."""
    config = MockConfig()
    launcher = RuntimePluginLauncher(config, "WORKER", 4, 1)

    # Create mock processes
    mock_proc1 = Mock()
    mock_proc1.poll = Mock(return_value=None)  # Still running
    mock_proc2 = Mock()
    mock_proc2.poll = Mock(return_value=0)  # Already exited

    launcher.plugin_processes = [mock_proc1, mock_proc2]

    launcher.stop_plugins()

    # Should terminate running process
    mock_proc1.terminate.assert_called_once()
    # Should not terminate exited process
    mock_proc2.terminate.assert_not_called()


@patch("lmcache.v1.plugin.runtime_plugin_launcher.atexit.register")
def test_atexit_registration(mock_register):
    """Test that cleanup handler is registered at initialization."""
    config = MockConfig()

    launcher = RuntimePluginLauncher(config, "WORKER", worker_count=4, worker_id=1)

    # Should register stop_plugins with atexit
    mock_register.assert_called_once_with(launcher.stop_plugins)
