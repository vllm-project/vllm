## Cutlass Kernel Generator and Benchmark Sweeps

#### Basic Idea
 - Expose a C++ interface for the function to benchmark. The interface must be
   templated with the hyper-parameters we desire to sweep over.
 - Generate .cu files using jinja templates that use the exposed interface.
   Look at `scaled_mm_c3x.jinja`
 - Generate torch bindings for the functions in the .cu files.
 - Build vllm to include all the generated .cu files. Look at `nm_cutlass_c.cmake`
 - Run the benchmarking script to sweep over problem shapes and all the generated
   cutlass kernels. Look at `benchmarks/cutlass_benchmarks/bench_v2.py`

#### Important Files
 - scaled_mm_c3x.jinja / simple_gemm_c3x.jinja : Jinja templated files for functions to generate.
 - scaled_mm_c3x_fnprototype.jinja / simple_gemm_c3x_fnprototype.jinja : Jinja templated files for the C++ function declarations.
 - generator_types.py : This file contains all the information regarding the function type we intend to generate.
        For example, at the time of writing, we have ScaledMMGeneratorType and SimpleGemmGeneratorType.
        The ScaledMMGeneratorType points to the correct jinja templates to use and also defines the
        correct torch biniding `ops.impl` and `ops.def` string. This is where we register new GeneratorTypes
        if we add more function-generators in the future.
 - autogen_manifest.py : Defines hyper-parameter sets.
 - kernel_generator.py : All utilities that are responsible for filling out the jinja templates
        based on the given set of hyper-parameter args.
 - generator.py : Bridges autogen_manifest.py and kernel_generator.py. This is the `main` driver
        scripts that we use to generate kernels.
 - kernel_compiler.py : Not all sets of hyperparameters are valid. The KernelCompiler, attempts an
        nvcc compile on the generated kernel file and kernel_generator/generator accepts/rejects
        the generated kernel based this compilation status.

#### Adding a new function to generate

##### Step 1
    - Like mentioned before, expose a C++ interface for the function to generate. The interface
    must be templated with the hyper-parameters we desire to sweep over.

##### Step 2
    - Create jinja templates.
        1. Create a jinja template file that is representative of the kernel we wish to generate. 
        2. Create a separate jinja template file that has the function declaration.
    - Refer to `scaled_mm_c3x.jinja` and `scaled_mm_c3x_fnprototype.jinja`

##### Step 3
    - Create a GeneratorType in generator_types.py
    - The GeneratorType is the datastructure that communicates,
        1. What jinja template files to use
        2. What is the torch_bindings `ops.def` and `ops.impl` arguments
    - Refer to ScaledMMGeneratorType

##### Step 4
    - In autogen_manifest, create a list of hyper-parameter sets that are to be translated into kernel files.
    - Look at the construction of Cutlass3xArgsTest in autogen_manifest.py

##### Commands to generate kernels:
    - Example command:
    python3 csrc/quantization/cutlass_w8a8/generator/generator.py --generator-type scaled_mm --vllm-root-dir ${HOME}/code/nm-vllm-ent/nm-vllm-ent/ --py-venv-dir ${HOME}/code/nm-vllm-ent/nm-vllm-ent/vllm-test --cuda-dir /usr/local/cuda-12.5 --cutlass-args-list Cutlass3xArgsTest

    Here: 
        - --generator-type : The description of the desired GeneratorType in generator_types.py
        - --vllm-root-dir : The root-dir of your vllm project
        - --py-venv-dir : The root-dir of your python environment
        - --cuda-dir : cuda dir to use
        - --cutlass-args-list : the name of the list of hyper-parameter sets that you created in autogen_manifest.py

    Expectations:
     The generator attempts to generate one kernel for every hyper-parameter set.
        - The generator looks generates the kernel file
        - The generator attempts to compile the generated kernel file
        - If compilation succeeds, it keeps the generated kernel file. Deletes it otherwise.

    The generator records the status of the compilation for each kernel it tries to compile. If some kernel is known to 
    have succeeded in a previous run, it simply generates it and doesnot attempt a re-compile.

##### Commands to build
    - The normal vllm build command should work.
    - i.e. either `pip3 install -e .` or `python3 setup.py --build_ext --inplace`
    Expectation:
        Compilation should be successful and you should see .so files like, `_nm_cutlass_*_C.so` in the vllm folder

##### How to benchmark
The benchmarking scripts have been updated to grab all the auto-generated cutlass kernels. Look at 
`get_autogen_functions` in `benchmarks/cutlass_benchmarks/bench_v2.py`.

Example command:
python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 --with-arg-pool 32 --with-cuda-graph 32 square_bench --dim-start 128 --dim-end 256 --dim-increment 128

Expectations:
    You should see output similar to, 
     ```
     attempting import vllm._nm_cutlass_0_C
     #autogen functions found 3
    Bench autogen autogen_scaled_mm_90_64x64x32_1x1x1_KernelTmaWarpSpecializedFP8FastAccum_TmaWarpSpecializedCooperative_PersistentScheduler_kGemm_float_fp8
    Bench autogen autogen_scaled_mm_90_64x64x32_1x1x1_KernelTmaWarpSpecializedFP8FastAccum_TmaWarpSpecialized_PersistentScheduler_kGemm_float_fp8
    Bench autogen autogen_scaled_mm_90_64x64x32_1x1x1_KernelTmaWarpSpecializedPingpongFP8FastAccum_TmaWarpSpecialized_PersistentScheduler_kGemm_float_fp8
     ```

##### Benchmark Heatmaps and Optimal Kernel Set Selection
Typically a hyper-parameter sweep produces 100s of kernels. It could be hard to read the terminal outputs
of benchmarking scripts. The w8a8_benchmarks.py script when used with the model_bench command, produces
a pickle file that contains the benchmark information for all the {kernel, gemm-shape} pairs benchmarked.

###### Kernel Selection Problem
When we run a hyper-parameter sweep, we are interested in finding a minimal a set of kernels that is the
optimal for the gemm-shapes benchmarked. `tools/select_kernels.py` solves this optimization problem.

Example:
 python3 select_kernels.py --input-pkl ./model_bench-torch.float8_e4m3fn-1729989172.pkl --min-gemm-efficiency 0.98

 This example invocation of the select_kernels.py script,
  - Reads the input pickle file and gathers the benchmark information of all the {kernel, gemm-shape} pairs.
  - Normalizes the benchmark information with respect to gemm shapes. i.e. the best performing
    kernel for some gemm-shape is given a value of 1.0. A kernel with a value of `x` ( `x` < 1.0)
    indicates that that kernel's performance is `x` times that of the optimal kernel.
  - The script ignores all the {kernel, gemm-shape} pairs where the kernel efficiency is < min_gemm_efficiency.
    In this case the script only considers the {kernel, gemm-shape} pairs where the normalized value
    is in range [0.98, 1.0]
  - The script then determines the optimal and minimal kernel set.

###### Visualization problem
Reading the w8a8_benchmarks.py terminal output can get overwhelming. The script `tools/heatmap.py`
consumes a model_bench pickle file and produces a heatmap for better consumption of the results.

Example:
  python3 heatmap.py --input-pkl ./model_bench-torch.float8_e4m3fn-1730295961-selected.pkl --plot-all-ops

  Normalizes all the {kernel, gemm-shape} information in the model_bench pickle file (refer to "Kernel Selection Problem"
  for how the data is normalized). and renders the normalized benchmark information as a heatmap.

Example:
  python3 heatmap.py --input-pkl ./model_bench-torch.float8_e4m3fn-1730295961-selected.pkl --select-kernels

  Effectively runs select_kernel.py on the input pkl file and renders the selected kernels as heatmap.







tools/select_kernel.py :  




