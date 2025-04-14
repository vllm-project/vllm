# SPDX-License-Identifier: Apache-2.0
"""
This file is a (slightly adapted) copy of 
https://github.com/IBM/triton-dejavu/blob/main/triton_dejavu/jit_cache.py 
(copied here to reduce external dependencies).

The `jitcache` reduces the launch overhead of triton kernels to 30-40us.

Details see: https://github.com/IBM/triton-dejavu

Authors:
- Burkhard Ringlein <ngl@zurich.ibm.com>

"""

import inspect
import time

from triton import KernelInterface
from triton import __version__ as triton_version
from triton.runtime.autotuner import OutOfResources
from triton.runtime.driver import driver

from vllm.logger import init_logger

logger = init_logger(__name__)

__print_name__ = "vllm.triton_utils.jit_cache"


class CacheLock:

    def __init__(self, id="unknown"):
        self.is_locked = False
        self.id = id

    def lock(self):
        self.is_locked = True
        logger.debug("JitCache lock '%s' is LOCKED.", self.id)

    def unlock(self):
        self.is_locked = False
        logger.debug("JitCache lock '%s' is UNLOCKED.", self.id)


# to provide a global lock
global_cache_lock = CacheLock("global")


class PreparedKernel:

    def __init__(
        self,
        grid_obj,
        grid_example,
        cache_launch_grid,
        kernel,
        launch_metadata,
        launch_enter_hook,
        launch_exit_hook,
        non_const_arg_names,
        assume_const_vals_dict,
        update_only_arg_names,
        cache_key,
        device,
    ):
        self.grid_obj = grid_obj
        self.grid_is_callable = callable(grid_obj)
        self.grid_size = len(
            grid_example
        )  # grid_example is always not callable, so we need both
        self.cache_launch_grid = cache_launch_grid
        self.concrete_grid = None
        if cache_launch_grid:
            grid_0 = grid_example[0]
            grid_1 = grid_example[1] if self.grid_size > 1 else 1
            grid_2 = grid_example[2] if self.grid_size > 2 else 1
            self.concrete_grid = (grid_0, grid_1, grid_2)
        self.kernel = kernel
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook

        self.non_const_arg_names = non_const_arg_names
        self.non_const_vals_lst = []
        self.update_args_index = {}
        for i, arg_n in enumerate(self.non_const_arg_names):
            if arg_n in update_only_arg_names:
                self.update_args_index[arg_n] = i
                self.non_const_vals_lst.append('dummy_value')
            else:
                self.non_const_vals_lst.append(assume_const_vals_dict[arg_n])

        self.device = device
        self._init_handles()
        self.cache_key = cache_key

    def _init_handles(self):
        """
        more or less redo what CompiledKernel._init_hanles is doing
        (c.f. triton/python/triton/runtime/compiler.py:379)
        """
        self.run = driver.active.launcher_cls(self.kernel.src,
                                              self.kernel.metadata)
        # check once and not again
        self.dev_max_shared = driver.active.utils.get_device_properties(
            self.device)["max_shared_mem"]
        if self.kernel.metadata.shared > self.dev_max_shared:
            raise OutOfResources(self.metadata.shared, self.dev_max_shared,
                                 "shared memory")
        self.module, self.function, self.n_regs, self.n_spills = (
            driver.active.utils.load_binary(
                self.kernel.name,
                self.kernel.kernel,
                self.kernel.metadata.shared,
                self.device,
            ))

    def __call__(self, *args, **kwargs):
        assert len(args) == 0

        # non_constsexpr_vals = []
        # # order is always the same...
        # for arg_n in self.non_const_arg_names:
        #     non_constsexpr_vals.append(kwargs[arg_n])
        for arg_n, idx in self.update_args_index.items():
            self.non_const_vals_lst[idx] = kwargs[arg_n]

        if self.cache_launch_grid:
            grid_0, grid_1, grid_2 = self.concrete_grid
        else:
            if self.grid_is_callable:
                grid = kwargs["grid"](kwargs)
            else:
                grid = kwargs["grid"]
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

        stream = driver.active.get_current_stream(self.device)

        return self.run(
            grid_0,
            grid_1,
            grid_2,
            stream,
            self.function,
            self.kernel.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            # *non_constsexpr_vals,
            *self.non_const_vals_lst,
        )

    def get_key(self):
        return self.cache_key


class JitCache(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        check_keys,
        cache_lock: CacheLock,
        cache_launch_grid=False,
        assume_const=None,
    ):
        # we depend on the triton version, right now, 3.0 -- 3.2 are supported
        assert (int(triton_version.split(".")[0]) == 3
                and int(triton_version.split(".")[1]) <= 2)
        self.arg_names = arg_names
        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn
        self.cache_lock = cache_lock
        self.cache_launch_grid = cache_launch_grid
        self.dynamic_mode = False
        self.run = self._run_static
        if self.cache_lock is None:
            self.dynamic_mode = True
            self.run = self._run_dynamic
        self.check_keys = check_keys
        self.assume_const = assume_const
        self.kernel_cache = {}

        def calc_cache_index(kwargs):
            cache_key = ""
            for c_arg_name in check_keys:
                cache_key += str(kwargs[c_arg_name])
            return cache_key

        self.cache_index_func = calc_cache_index
        if len(check_keys) == 0:
            self.cache_index_func = lambda ignore: "_default_"

    def _get_prepared_kernel(self, *args, **kwargs) -> PreparedKernel:
        """
        more or less redo what JITFunction.run is doing
        (c.f. triton/python/triton/runtime/jit.py:565)
        """

        kwargs["warmup"] = True
        compile_start = time.time()
        kernel = self.fn.run(*args, **kwargs)
        compile_end = time.time()

        const_arg_names = []
        non_const_arg_names = []
        for p in self.fn.params:
            if p.is_constexpr or p.is_const:
                const_arg_names.append(p.name)
            else:
                non_const_arg_names.append(p.name)
        if any(x in self.check_keys for x in non_const_arg_names):
            raise RuntimeError(
                f"[{__print_name__}] ERROR: check_keys must only contain"
                "parameters marked as tl.constexpr (non-constants will be "
                "updated in all cases).")
        if self.assume_const:
            if any(x in self.assume_const for x in const_arg_names):
                raise RuntimeError(
                    f"[{__print_name__}] ERROR: assume_const must only contain"
                    "parameters NOT marked as tl.constexpr.")
            update_only_arg_names = [
                arg_n for arg_n in non_const_arg_names
                if arg_n not in self.assume_const
            ]
            assume_const_vals_dict = {
                arg_n: kwargs[arg_n]
                for arg_n in non_const_arg_names if arg_n in self.assume_const
            }
        else:
            update_only_arg_names = non_const_arg_names
            assume_const_vals_dict = {}

        (
            bound_args,
            sig_and_spec,
            constexpr_vals,
            non_constexpr_vals,
            excess_kwargs,
        ) = self.fn.binder(*args, **kwargs)
        bind_end = time.time()

        if callable(kwargs["grid"]):
            grid = kwargs["grid"](kwargs)
        else:
            grid = kwargs["grid"]

        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream,
                                                 *non_constexpr_vals)

        prepared_kernel = PreparedKernel(
            kwargs["grid"],
            grid,
            self.cache_launch_grid,
            kernel,
            launch_metadata,
            self.fn.CompiledKernel.launch_enter_hook,
            self.fn.CompiledKernel.launch_exit_hook,
            non_const_arg_names,
            assume_const_vals_dict,
            update_only_arg_names,
            self.cache_index_func(kwargs),
            device,
        )

        wrapper_end = time.time()
        compile_time = compile_end - compile_start
        bind_time = bind_end - compile_end
        wrapper_time = wrapper_end - bind_end

        logger.debug(
            "JIT compilation took %.2fs, binding %.2fs, wrapper %.2fs.",
            compile_time,
            bind_time,
            wrapper_time,
        )

        return prepared_kernel

    def _run_static(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs,"
                "len(args) must be 0.")
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        # print(f"my lock: {self.cache_lock.is_locked}")
        if not self.cache_lock.is_locked:
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                assert type(kwargs[key]) in [int, bool, float]
            prepared_kernel = self._get_prepared_kernel(*args, **kwargs)
            if prepared_kernel.get_key() in self.kernel_cache:
                logger.debug(
                    "WARNING: Kernel variant already cached, will override "
                    "(cache lock is not locked). "
                    "This could mean that the given check_keys are ambiguous "
                    "(or the same call was already executed).")
            self.kernel_cache[prepared_kernel.get_key()] = prepared_kernel

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError as e:
            logger.debug(
                "Key %s not in cache. Current cache %s",
                str(self.cache_index_func(kwargs)),
                str(list(self.kernel_cache.keys())),
            )
            raise e

        return kernel_variant(*args, **kwargs)

    def _run_dynamic(self, *args, **kwargs):
        # we only support kwargs
        if len(args) != 0:
            raise RuntimeError(
                f"[{__print_name__}] ERROR: The JITCache only supports kwargs, "
                "len(args) must be 0.")
        # assert no config pre-hook
        assert "pre_hook" not in kwargs or kwargs["pre_hook"] is None

        try:
            kernel_variant = self.kernel_cache[self.cache_index_func(kwargs)]
        except KeyError:
            logger.debug(
                "Key %s  not in cache, compiling...\n"
                "Current cache: %s",
                str(self.cache_index_func(kwargs)),
                str(list(self.kernel_cache.keys())),
            )
            # we only support int, bool, float as cache index
            for key in self.check_keys:
                assert type(kwargs[key]) in [int, bool, float]
            kernel_variant = self._get_prepared_kernel(*args, **kwargs)
            self.kernel_cache[kernel_variant.get_key()] = kernel_variant

        return kernel_variant(*args, **kwargs)


def jitcache(
    check_keys,
    cache_lock=None,
    cache_launch_grid=False,
    assume_const=None,
):
    """
    Decorator for caching a :code:`triton.jit`'d function.

    :param check_keys: The list of tl.constexpr that are used to index
                       the cache. Only types int, bool, float are supported.
    :type check_keys: list[str]
    :param cache_lock: The CacheLock used for this JitCache.
    :type cache_lock: CacheLock
    :param chache_launch_grid: Indicate if the launch grid size is static and
                               should be cached (False by default).
    :type cache_launch_grid: bool
    :param assume_const: A list of parameters that are NOT marked as
                         tl.constexpr but should be treated as constants in
                         this kernel launch.
    :param assume_const: list[str]
    """

    def decorator(fn):
        return JitCache(
            fn,
            fn.arg_names,
            check_keys,
            cache_lock,
            cache_launch_grid,
            assume_const,
        )

    return decorator
