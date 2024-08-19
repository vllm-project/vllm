export MAX_JOBS=6 
echo "MAX_JOBS is set to: $MAX_JOBS" 
export CUDA_HOME=/usr/local/cuda 
export PATH="${CUDA_HOME}/bin:$PATH" 
nvcc --version # verify that nvcc is in your PATH 
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOMEa