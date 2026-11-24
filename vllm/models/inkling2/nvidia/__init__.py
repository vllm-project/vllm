# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 CUDA package: the CUDA-only features and kernels, and the CUDA
entry modules `nvidia.model` / `nvidia.loop`.

The modules that moved to `common/` stay importable under their old paths
here (`_MOVED`): importing one returns the `common/` module object itself,
so imports, private names and monkeypatches through the old path keep
working. There are no files at those paths, so git follows the renames and
in-flight changes to the old files rebase onto the moved ones."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

# Old module (relative to this package) -> its module under inkling2/.
_MOVED = {
    "attention": "common.attention",
    "loop": "common.loop",
    "model": "common.model",
    "moe": "common.moe",
    "sconv": "common.sconv",
    "ops.attention_sm90": "common.ops.attention_sm90",
    "ops.attn_res": "common.ops.attn_res",
    "ops.conv_mix": "common.ops.conv_mix",
    "ops.e4m3": "common.ops.e4m3",
    "ops.linear": "common.ops.linear",
    "ops.loop": "common.ops.loop",
    "ops.norm": "common.ops.norm",
    "ops.pdl": "common.ops.pdl",
    "ops.sconv": "common.ops.sconv",
}
_PARENT = __name__.rsplit(".", 1)[0]


class _MovedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves `<this package>.<old>` to the moved module."""

    def find_spec(self, fullname, path=None, target=None):
        old = fullname.removeprefix(f"{__name__}.")
        if old == fullname or old not in _MOVED:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return None

    def exec_module(self, module) -> None:
        old = module.__name__.removeprefix(f"{__name__}.")
        # The import system returns what `sys.modules` holds afterwards.
        sys.modules[module.__name__] = importlib.import_module(
            f"{_PARENT}.{_MOVED[old]}"
        )


if not any(isinstance(finder, _MovedModules) for finder in sys.meta_path):
    sys.meta_path.append(_MovedModules())
