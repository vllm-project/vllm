import enum
import functools
import os
import subprocess
import sys
import torch
from triton_kernels.numerics import MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5


def assert_equal(ref, tri):
    if isinstance(ref, torch.Tensor):
        assert torch.all(ref == tri)
    else:
        assert ref == tri


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if ref.numel() == 0:
        return

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert ref.shape == tri.shape, f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(inf_mask_ref, inf_mask_tri), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print("%s maximum relative error = %s (threshold = %s)" % (description, max_err, maxtol))
        print("%s RMS relative error = %s (threshold = %s)" % (description, rms_err, rmstol))

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print("%d / %d mismatched elements (shape = %s) at coords %s" %
              (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist()))

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[tuple(bad_idxs)].cpu())
        print("tri values: ", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


class ComputeSanitizerTool(enum.Enum):
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"
    INITCHECK = "initcheck"


def compute_sanitizer(**target_kwargs):
    """
    Decorator to run a test with compute sanitizer enabled and pytorch caching allocator disabled,
    to expose potential memory access errors.
    This decorator requires the `request` fixture to be present.
    If `run_sanitizer` argument is present and set to False, the sanitizer is not run.
    Running tests under compute sanitizer requires launching subprocess and is slow,
    so use sparingly
    """

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            if os.environ.get("SKIP_COMPUTE_SANITIZER") == "1":
                test_fn(*args, **kwargs)
                return

            import psutil

            if target_kwargs.pop("clear_torch_cache", False):
                # If we don't pop clear_torch_cache, it won't pass
                # target_kwargs.items() <= kwargs.items() condition below.
                torch.cuda.empty_cache()
            tools_to_check = target_kwargs.pop("tools_to_check", [ComputeSanitizerTool.MEMCHECK])
            assert isinstance(tools_to_check, list), f"{tools_to_check=}"
            assert all(tool in ComputeSanitizerTool for tool in tools_to_check), (
                f"{(tool for tool in tools_to_check if tool not in ComputeSanitizerTool)=}")

            ppid_name = psutil.Process(os.getppid()).exe()
            run_compute_sanitizer = target_kwargs.items() <= kwargs.items()
            if "run_sanitizer" in kwargs:
                run_compute_sanitizer &= kwargs["run_sanitizer"]
            if run_compute_sanitizer and "compute-sanitizer" not in ppid_name:
                for tool in tools_to_check:
                    path = os.path.realpath(test_fn.__globals__["__file__"])
                    # get path of current file
                    env = {
                        "PATH": os.environ["PATH"],
                        "PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
                        "TORCH_SHOW_CPP_STACKTRACES": "1",
                        "CUDA_LAUNCH_BLOCKING": "1",
                    }
                    if "CUDA_VISIBLE_DEVICES" in os.environ:
                        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
                    assert "request_fixture" in kwargs, (
                        "memcheck'ed test must have a (possibly unused) `request` fixture")
                    test_id = kwargs["request_fixture"].node.callspec.id
                    cmd = f"{path}::{test_fn.__name__}[{test_id}]"
                    cmd = [
                        "compute-sanitizer",
                        "--target-processes=application-only",
                        "--destroy-on-device-error=context",
                        f"--tool={tool.value}",
                        sys.executable,
                        "-m",
                        "pytest",
                        "-vsx",
                        cmd,
                    ]
                    for opt in ["--update_checksum", "--ignore_checksum_error"]:
                        if opt in sys.argv:
                            cmd.append(opt)
                    out = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                    sanitizer_ok = "ERROR SUMMARY: 0 errors" in str(
                        out.stdout) or "RACECHECK SUMMARY: 0 hazards displayed" in str(out.stdout)
                    test_output = out.stdout
                    if type(test_output) is bytes:
                        test_output = test_output.decode()

                    fail = False
                    if not sanitizer_ok:
                        print("compute-sanitizer returned an error")
                        fail = True
                    elif out.returncode != 0:
                        print(
                            "The test failed due to some other reason: consider running without compute-sanitizer to verify."
                        )
                        print(f"{out.returncode=}")
                        fail = True

                    if fail:
                        print("*****************************************************")
                        print("******************** TEST OUTPUT ********************")
                        print("*****************************************************")
                        print(test_output)
                        print("*****************************************************")
                        print("****************** TEST OUTPUT END ******************")
                        print("*****************************************************")
                        assert None
            else:
                test_fn(*args, **kwargs)

        return wrapper

    return decorator


def compute_actual_scale(x, dtype):
    max_finite = {
        torch.float8_e5m2: MAX_FINITE_FLOAT8E5,
        torch.float8_e4m3fn: MAX_FINITE_FLOAT8E4NV,
        torch.float8_e4m3fnuz: MAX_FINITE_FLOAT8E4B8,
    }[dtype]
    return x.abs().max() / max_finite
