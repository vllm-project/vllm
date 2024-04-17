import multiprocessing
import tempfile

from vllm.utils import update_environment_variables


def target_fn(env, filepath):
    update_environment_variables(env)
    import os
    exit_code = os.system(f"ldd {filepath}")
    if exit_code != 0:
        exit(-1)
    from vllm.distributed.device_communicators.pynccl import ncclGetVersion
    ncclGetVersion()


def test_library_file():
    from vllm.distributed.device_communicators.pynccl import so_file
    with open(so_file, 'rb') as f:
        content = f.read()
    try:
        # corrupt the library file, should raise an exception
        with open(so_file, 'wb') as f:
            f.write(content[:len(content) // 2])
        p = multiprocessing.Process(target=target_fn, args=({}, so_file))
        p.start()
        p.join()
        assert p.exitcode != 0

        # move the library file to a tmp path
        # test VLLM_NCCL_SO_PATH
        fd, path = tempfile.mkstemp()
        with open(path, 'wb') as f:
            f.write(content)
        p = multiprocessing.Process(target=target_fn,
                                    args=({
                                        "VLLM_NCCL_SO_PATH": path
                                    }, path))
        p.start()
        p.join()
        assert p.exitcode == 0
    finally:
        with open(so_file, 'wb') as f:
            f.write(content)
