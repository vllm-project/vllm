# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        default="configs/models-turboquant.txt",
        help="File containing list of config files to test",
    )


def pytest_generate_tests(metafunc):
    if "config_filename" in metafunc.fixturenames:
        config_list_file = metafunc.config.getoption("--config-list-file")

        config_list_path = Path(config_list_file)
        if not config_list_path.is_absolute():
            test_dir_path = Path(__file__).parent / config_list_file
            if test_dir_path.exists():
                config_list_path = test_dir_path
            else:
                config_list_path = Path.cwd() / config_list_file

        config_files = []
        if config_list_path.exists():
            config_dir = config_list_path.parent
            with open(config_list_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        config_path = config_dir / line
                        if config_path.exists():
                            config_files.append(config_path)

        if config_files:
            metafunc.parametrize(
                "config_filename",
                config_files,
                ids=[cf.stem for cf in config_files],
            )
