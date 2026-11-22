# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--config-list-file",
        default="configs/models-small.txt",
        help="File containing list of config files to test",
    )
    parser.addoption(
        "--drift-out",
        default="logprob-drift",
        help="Directory the runs are recorded to, one subdirectory per config; "
        "runs already there are kept",
    )
    parser.addoption(
        "--drift-reference",
        help="A --drift-out directory of the reference build to compare against",
    )
    parser.addoption(
        "--drift-runs",
        type=int,
        default=2,
        help="Runs of the reference build to record (default 2); the candidate's "
        "test asks for more when it needs them",
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

        config_files = []
        if config_list_path.exists():
            for line in config_list_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    config_files.append(config_list_path.parent / line)
        else:
            print(f"Config list file not found: {config_list_path}")

        if config_files:
            metafunc.parametrize(
                "config_filename",
                config_files,
                ids=[config_file.stem for config_file in config_files],
            )
