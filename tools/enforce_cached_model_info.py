# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2025 vLLM Team
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
"""
Generates a cached dictionary with _ModelInfo data argumnents.
Check if the cached dictionary is up to date with files.
"""

import argparse
import ast
import hashlib
import importlib
import pprint
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path

_CACHED_FILE = "_cached_model_info.py"


def _build_cache(filenames: list[str]) -> tuple[dict, list[str]]:
    from vllm.model_executor.models.registry import _VLLM_MODELS, _ModelInfo

    try:
        from vllm.model_executor.models._cached_model_info import (
            _CACHED_MODEL_INFO)
    except Exception:
        _CACHED_MODEL_INFO = {}

    # build cache for modified model files only
    # if no modified files were passed, build the cache for all of them
    module_names = set()
    for filename in filenames:
        if not filename.endswith(_CACHED_FILE):
            module_names.add(filename.replace("/", ".").removesuffix(".py"))

    if len(module_names) > 0:
        print(f"Build cache for modules: {', '.join(module_names)}")
    else:
        print("Build cache for all modules")

    list_errors = []
    new_info_dict = {}
    modules = {}
    for _, (mod_relname, cls_name) in _VLLM_MODELS.items():
        try:
            cached_value = _CACHED_MODEL_INFO.get(cls_name)
            module_name = f"vllm.model_executor.models.{mod_relname}"
            # if there is no cached class or the are no passed files
            # or this file was passed, regenerate this module
            if (cached_value is None or len(module_names) == 0
                    or module_name in module_names):
                if module_name not in modules:
                    mod = importlib.import_module(module_name)
                    with open(mod.__file__, "rb") as f:
                        md5_hash = hashlib.md5(f.read()).hexdigest()
                    modules[module_name] = {
                        "module": mod,
                        "md5hash": md5_hash,
                    }

                md5_hash = modules[module_name]["md5hash"]
                mod = modules[module_name]["module"]
                modelinfo = _ModelInfo.from_model_cls(getattr(mod, cls_name))
                mi_dict = {}
                for field in fields(modelinfo):
                    mi_dict[field.name] = getattr(modelinfo, field.name)

                new_info_dict[cls_name] = {
                    "module": mod_relname,
                    "md5hash": md5_hash,
                    "modelinfo": mi_dict,
                }
            else:
                new_info_dict[cls_name] = cached_value

        except Exception:
            list_errors.append(traceback.format_exc())

    return new_info_dict, list_errors


def _compare_dicts(old_dict: dict, new_dict: dict) -> bool:
    length_old = len(old_dict)
    length_new = len(new_dict)
    if length_old != length_new:
        print(
            f"Cached dict length changed from '{length_old}'  to '{length_new}'"
        )
        return False

    for old_key, old_value in old_dict.items():
        new_value = new_dict.get(old_key)
        if new_value is None:
            print(f"New cached dict removed key '{old_key}'")
            return False

        if type(old_value) is not type(new_value):
            print(f"New cached dict key '{old_key}' value type "
                  f"changed from '{type(old_value)}' to "
                  f"'{type(new_value)}'")
            return False

        if isinstance(new_value, dict):
            if not _compare_dicts(old_value, new_value):
                return False
            continue

        if old_value != new_value:
            print(f"New cached dict key '{old_key}' value "
                  f"changed from '{old_value}' to '{new_value}'")
            return False

    return True


def _compare_with_existent_cache(new_model_dict: dict) -> bool:
    try:
        from vllm.model_executor.models._cached_model_info import (
            _CACHED_MODEL_INFO)
    except Exception:
        _CACHED_MODEL_INFO = {}

    return _compare_dicts(_CACHED_MODEL_INFO, new_model_dict)


def _print_errors(list_errors: list[str]) -> None:
    print("\nERRORS:", file=sys.stderr)
    for msg_error in list_errors:
        print(f"\n{msg_error}", file=sys.stderr)

    print("", file=sys.stderr)


def _build(filenames: list[str]):
    model_info_dict, list_errors = _build_cache(filenames)
    if len(list_errors) > 0:
        _print_errors(list_errors)
        return 1

    if _compare_with_existent_cache(model_info_dict):
        print("Model Info dictionary didn't change.")
        return 0

    content = "# SPDX-License-Identifier: Apache-2.0\n"
    content += (
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    )
    content += "# Copyright 2023-2025 vLLM Team\n"
    content += (
        '# Licensed under the Apache License, Version 2.0 (the "License");\n')
    content += (
        "# You may not use this file except in compliance with the License.\n")
    content += "# You may obtain a copy of the License "
    content += "at http://www.apache.org/licenses/LICENSE-2.0\n"
    content += "#\n"
    content += (
        f"# This file was automatically generated by '{Path(__file__).name}'\n"
    )
    content += "# It should not be changed manually.\n\n"

    content += "# yapf: disable\n"
    content += "# ruff: noqa\n\n"

    content += "_CACHED_MODEL_INFO = \\\n"
    content += pprint.pformat(model_info_dict, indent=1, sort_dicts=False)

    model_info_path = Path(__file__).parent.parent
    model_info_path = (model_info_path.joinpath("vllm").joinpath(
        "model_executor").joinpath("models"))
    model_info_file = model_info_path.joinpath(_CACHED_FILE)

    model_info_file.write_text(content, encoding="utf-8")

    print(f"Updated file {model_info_file}")

    return 0


def _compare_dict_and_files() -> int:
    models_path = Path(__file__).parent.parent
    models_path = (models_path.joinpath("vllm").joinpath(
        "model_executor").joinpath("models"))

    model_info_path = models_path.joinpath(_CACHED_FILE)
    if not model_info_path.exists():
        print(f"Cached Model file: {model_info_path} doesn't exist")
        return 1

    contents = model_info_path.read_text(encoding="utf-8")
    dict_literal_string = contents.split("=", 1)[1].strip()
    model_dict = ast.literal_eval(dict_literal_string)

    changed_files = []
    cached_model_files = {}
    for _, model_info_dict in model_dict.items():
        module = model_info_dict["module"]
        md5_hash = cached_model_files.get(module)
        if md5_hash is None:
            file_path = models_path.joinpath(module + ".py")
            if not file_path.exists():
                print(f"Module file {file_path} not found")
                continue

            with open(file_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
            cached_model_files[module] = md5_hash

        if md5_hash != model_info_dict["md5hash"]:
            changed_files.append(module + ".py")

    if len(changed_files) > 0:
        print(f"Model file(s): {', '.join(changed_files)} "
              "out of sync with cached dictionary")
        return 1

    print("Model files in sync with cached dictionary")
    return 0


def _main():
    start_time = time.perf_counter()
    try:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Generate model info dictionary file",
        )
        parser.add_argument("filenames",
                            nargs="*",
                            help="List of filenames to process")
        parser.add_argument(
            "-b",
            "--build",
            default=False,
            action="store_true",
            help="Builds _ModelInfo dictionary",
        )
        args = parser.parse_args()

        if args.build:
            print("Start Building cache process ...")
            return _build(args.filenames)

        print("Start verifying cache process ...")
        return _compare_dict_and_files()
    finally:
        elapsed_time = time.perf_counter() - start_time
        print(f"Script took {elapsed_time:7f} secs")


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
