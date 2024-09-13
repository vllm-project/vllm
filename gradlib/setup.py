import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.hipify import hipify_python

this_dir = os.path.dirname(os.path.abspath(__file__))
#gpus = subprocess.check_output("/opt/rocm/bin/rocminfo").decode(
#           'UTF-8').split('\n')
#gpus = list(set([re.search('(gfx94.)', g).group(0)
#            for g in gpus if 'gfx94' in g]))
gpus = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
#gpus = ['gfx90a','gfx940']
extra_args = ["--offload-arch=" + g for g in gpus]

#sets_rocm_pytorch = False
maj_ver, min_ver, *_ = torch.__version__.split('.')
if int(maj_ver) > 1 or (int(maj_ver) == 1 and int(min_ver) >= 5):
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = bool(torch.version.hip is not None
                           and ROCM_HOME is not None)

ext_modules = []

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, 'include', 'ATen',
                               'CUDAGenerator.h')):
    generator_flag = ['-DOLD_GENERATOR']

print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

version_ge_1_1 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
    version_ge_1_1 = ['-DVERSION_GE_1_1']
version_ge_1_3 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
    version_ge_1_3 = ['-DVERSION_GE_1_3']
version_ge_1_5 = []
if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
    version_ge_1_5 = ['-DVERSION_GE_1_5']
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

include_dirs = [os.path.join(this_dir, 'csrc')]

#if is_rocm_pytorch:
#    import shutil
#    with hipify_python.GeneratedFileCleaner(
#    keep_intermediates=True) as clean_ctx:
#        hipify_python.hipify(project_directory=this_dir,
#                             output_directory=this_dir, includes="csrc/*",
#				              show_detailed=True,
#                             is_pytorch_extension=True,
#                             clean_ctx=clean_ctx)

if not is_rocm_pytorch:
    ext_modules.append(
        CUDAExtension(name='gradlib',
                      sources=['grad_funcs.cu'],
                      extra_compile_args={
                          'cxx': [
                              '-O3',
                          ],
                          'nvcc': [
                              '-O3', '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              "--expt-relaxed-constexpr",
                              "-ftemplate-depth=1024",
                              '-gencode=arch=compute_70,code=sm_70',
                              '-gencode=arch=compute_80,code=sm_80',
                              '-gencode=arch=compute_80,code=compute_80'
                          ]
                      }))
elif is_rocm_pytorch:
    #if torch.__version__ <= '1.8':
    hipify_ver = [
        int(x) for x in torch.utils.hipify.__version__.split(".")
    ] if hasattr(torch.utils.hipify, "__version__") else [0, 0, 0]
    if hipify_ver < [1, 0, 0]:
        with hipify_python.GeneratedFileCleaner(
                keep_intermediates=True) as clean_ctx:
            hipify_python.hipify(project_directory=this_dir,
                                 output_directory=this_dir,
                                 includes="csrc/*",
                                 show_detailed=True,
                                 is_pytorch_extension=True,
                                 clean_ctx=clean_ctx)

        ext_modules.append(
            CUDAExtension(name='gradlib',
                          sources=['./csrc/hip/grad_funcs.hip'],
                          extra_compile_args={
                              'cxx': [
                                  '-O3',
                              ] + version_dependent_macros,
                              'nvcc': ['-O3'] + extra_args
                          }))
    else:
        #ext_modules.append(
        #            CUDAExtension(
        #                name='gradlib',
        #                sources=['./csrc/grad_funcs.cu'],
        #                include_dirs=include_dirs,
        #                # add additional libraries argument for hipblaslt
        #                libraries=['hipblaslt'],
        #                extra_compile_args={
        #                    'cxx': ['-O3',],
        #                    'nvcc':['-O3',
        #                    '-U__CUDA_NO_HALF_OPERATORS__',
        #                    '-U__CUDA_NO_HALF_CONVERSIONS__',
        #                    "-ftemplate-depth=1024"] + extra_args
        #                }
        #            )
        #        )
        ext_modules.append(
            CUDAExtension(
                name='rocsolidxgemm',
                sources=['./csrc/rocsolgemm.cu'],
                include_dirs=include_dirs,
                # add additional libraries argument for hipblaslt
                libraries=['rocblas'],
                extra_compile_args={
                    'cxx': [
                        '-O3',
                        '-DLEGACY_HIPBLAS_DIRECT=ON',
                    ],
                    'nvcc': [
                        '-O3',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        "-ftemplate-depth=1024",
                        '-DLEGACY_HIPBLAS_DIRECT=ON',
                    ] + extra_args
                }))
        ext_modules.append(
            CUDAExtension(
                name='hipbsolidxgemm',
                sources=['./csrc/hipbsolgemm.cu'],
                include_dirs=include_dirs,
                # add additional libraries argument for hipblaslt
                libraries=['hipblaslt'],
                extra_compile_args={
                    'cxx': [
                        '-O3',
                        '-DLEGACY_HIPBLAS_DIRECT=ON',
                    ],
                    'nvcc': [
                        '-O3',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        "-ftemplate-depth=1024",
                        '-DLEGACY_HIPBLAS_DIRECT=ON',
                    ] + extra_args
                }))

setup(name='gradlib',
      packages=['gradlib'],
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension})

# python setup.py build && cp build/lib*/gradlib* ../
