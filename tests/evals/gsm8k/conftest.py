# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import yaml

from vllm.platforms import current_platform


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--config-list-file",
        default="configs/models-small.yaml",
        help="File containing list of config files to test",
    )


def _list_keys() -> list[str]:
    """The .yaml list sections this machine runs: ``common``, its platform's
    device name and, on ROCm, its CDNA generation (``cdna3``, ``cdna4``)."""
    keys = ["common", current_platform.device_name]
    if current_platform.is_rocm():
        from vllm.platforms.rocm import get_cdna_version

        if cdna := get_cdna_version():
            keys.append(f"cdna{cdna}")
    return keys


def _config_names(config_list_path: Path) -> list[str]:
    """Config file names from a .txt list, or from a .yaml list's sections
    that this machine runs (see ``_list_keys``)."""
    if config_list_path.suffix in (".yaml", ".yml"):
        lists = yaml.safe_load(config_list_path.read_text()) or {}
        return [name for key in _list_keys() for name in lists.get(key) or []]
    lines = (line.strip() for line in config_list_path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def pytest_generate_tests(metafunc):
    """Generate test parameters from config files."""
    if "config_filename" in metafunc.fixturenames:
        config_list_file = metafunc.config.getoption("--config-list-file")

        # Handle both relative and absolute paths
        config_list_path = Path(config_list_file)
        if not config_list_path.is_absolute():
            # If relative, try relative to test directory first
            test_dir_path = Path(__file__).parent / config_list_file
            if test_dir_path.exists():
                config_list_path = test_dir_path
            else:
                # Try relative to current working directory
                config_list_path = Path.cwd() / config_list_file

        print(f"Looking for config list at: {config_list_path}")

        config_files = []
        if config_list_path.exists():
            # Determine config directory (same directory as the list file)
            config_dir = config_list_path.parent

            for name in _config_names(config_list_path):
                config_path = config_dir / name
                print(f"Checking config file: {config_path}")
                if config_path.exists():
                    config_files.append(config_path)
                    print(f"  ✓ Found: {config_path}")
                else:
                    print(f"  ✗ Missing: {config_path}")
        else:
            print(f"Config list file not found: {config_list_path}")

        # Generate test parameters
        if config_files:
            metafunc.parametrize(
                "config_filename",
                config_files,
                ids=[config_file.stem for config_file in config_files],
            )
        else:
            print("No config files found, test will be skipped")
