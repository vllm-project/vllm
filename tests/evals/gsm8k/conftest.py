# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--config-list-file",
        default="configs/models-small.txt",
        help="File containing list of config files to test",
    )
    parser.addoption("--tp-size", default=1, type=int, help="Tensor parallel size")


def pytest_generate_tests(metafunc):
    """Generate test parameters from config files."""
    if "config_filename" in metafunc.fixturenames:
        config_list_file = metafunc.config.getoption("--config-list-file")
        tp_size = metafunc.config.getoption("--tp-size")

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

            with open(config_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        config_path = config_dir / line
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
                ["config_filename", "tp_size"],
                [(config_file, int(tp_size)) for config_file in config_files],
                ids=[f"{config_file.stem}-tp{tp_size}" for config_file in config_files],
            )
        else:
            print("No config files found, test will be skipped")
