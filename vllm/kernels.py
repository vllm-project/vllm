# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import pkgutil
import sys

from vllm.logger import init_logger

logger = init_logger(__name__)


def _alias_modules(package_to_scan, new_parent_name):
    """
    Finds all modules in a given package and aliases them under a
    new parent.

    Args:
        package_to_scan: The package to scan for modules (e.g., the
                        'vllm_kernels' package).
        new_parent_name: The string name of the new parent package
                        (e.g., 'vllm').
    """

    package_path = package_to_scan.__path__
    package_name_prefix = package_to_scan.__name__ + '.'

    for _, module_name, _ in pkgutil.iter_modules(package_path,
                                                  prefix=package_name_prefix):
        original_module = importlib.import_module(module_name)

        # Alias the module under the new parent package
        short_name = module_name.split('.')[-1]
        new_module_name = f"{new_parent_name}.{short_name}"
        if new_module_name in sys.modules:
            # This should never happen, as we import vllm-kernels before
            # anything else. But in case it does, we do nothing to avoid
            # unexpected problems.
            logger.warning("%s exists in sys.modules, do nothing",
                           new_module_name)
            continue
        logger.info("import module %s from %s", new_module_name, module_name)
        sys.modules[new_module_name] = original_module
        setattr(sys.modules[new_parent_name], short_name, original_module)


try:
    import vllm_kernels
except ImportError as e:
    logger.debug(str(e), exc_info=True)
else:
    logger.info("Found vllm_kernels, importing modules")
    _alias_modules(vllm_kernels, "vllm")
