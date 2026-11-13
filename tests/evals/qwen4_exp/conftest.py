# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        required=True,
        help="File containing Qwen4Exp evaluation config files",
    )


def pytest_generate_tests(metafunc):
    if "config_filename" not in metafunc.fixturenames:
        return

    config_list_path = Path(metafunc.config.getoption("--config-list-file"))
    if not config_list_path.is_absolute():
        test_relative_path = Path(__file__).parent / config_list_path
        config_list_path = (
            test_relative_path
            if test_relative_path.exists()
            else Path.cwd() / config_list_path
        )

    config_dir = config_list_path.parent
    config_files = [
        config_dir / line.strip()
        for line in config_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [path for path in config_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing evaluation configs: {missing}")

    metafunc.parametrize(
        "config_filename",
        config_files,
        ids=[config_file.stem for config_file in config_files],
    )
