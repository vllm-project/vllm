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


def pytest_generate_tests(metafunc):
    """Generate test parameters from config files."""
    if "config_filename" in metafunc.fixturenames:
        config_list_path = Path(metafunc.config.getoption("--config-list-file"))
        if not config_list_path.is_absolute():
            test_dir_path = Path(__file__).parent / config_list_path
            if test_dir_path.exists():
                config_list_path = test_dir_path
            else:
                config_list_path = Path.cwd() / config_list_path

        lines = (line.strip() for line in config_list_path.read_text().splitlines())
        config_files = [
            config_list_path.parent / line
            for line in lines
            if line and not line.startswith("#")
        ]
        metafunc.parametrize(
            "config_filename",
            config_files,
            ids=[config_file.stem for config_file in config_files],
        )
