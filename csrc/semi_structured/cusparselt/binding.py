from torch.utils.cpp_extension import load
import os
import torch

base_path = __file__.replace("spmm.py", "")

if not os.path.exists(f"{base_path}/build"):
    os.makedirs(f"{base_path}/build")

if not os.path.exists(base_path + "/libcusparse_lt"):
    os.system(
    "wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.5.1.1-archive.tar.xz")
    os.system("tar -xf libcusparse_lt-linux-x86_64-0.5.1.1-archive.tar.xz")
    os.system(f"mv libcusparse_lt-linux-x86_64-0.5.1.1-archive {base_path}/libcusparse_lt")
    os.system("rm libcusparse_lt-linux-x86_64-0.5.1.1-archive.tar.xz")

pruner = load(name='pruner',
              sources=[f'{base_path}/spmm_backend.cpp',
                       f'{base_path}/spmm_backend.cu',
                       ],
              extra_cflags=[
                  f'-L{base_path}/libcusparse_lt/lib',
                  '-lcusparse',
                  '-lcusparseLt',
                  '-ldl'
              ],
              extra_cuda_cflags=[
                  f'-L{base_path}/libcusparse_lt/lib',
                  '-lcusparse',
                  '-lcusparseLt',
                  '-ldl'
              ],
              extra_ldflags=[
                  f'-L{base_path}/libcusparse_lt/lib',
                  '-lcusparse',
                  '-lcusparseLt',
                  '-ldl'
              ],
              extra_include_paths=[
                  base_path + '/libcusparse_lt/include'
              ],
              build_directory=f'{base_path}/build',
              with_cuda=True,
              verbose=False)

init_flag = pruner.init_cusparse_lt()
assert init_flag == 0, "Failed to initialize CuSparseLT"