export VERBOSE=1
#export NVCC_THREADS=8
export CUDA_HOME=/usr/local/cuda-12
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDACXX=/usr/local/cuda/bin/nvcc 
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=/usr/local/lib/python3.9/dist-packages/:$PYTHONPATH

#cd ./build
#/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_C -D_C_EXPORTS -I/root/vllm/csrc -I/root/vllm/build/_deps/cutlass-src/include -isystem /usr/include/python3.9 -isystem /usr/local/lib/python3.9/dist-packages/torch/include -isystem /usr/local/lib/python3.9/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include --expt-extended-lambda -O2 -g -Xptxas=-v --keep -std=c++17 --generate-code=arch=compute_80,code=[sm_80] -Xcompiler=-fPIC --expt-relaxed-constexpr -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_C.dir/csrc/attention/attention_kernels.cu.o -MF CMakeFiles/_C.dir/csrc/attention/attention_kernels.cu.o.d -x cu -c /root/vllm/csrc/attention/attention_kernels.cu -o CMakeFiles/_C.dir/csrc/attention/attention_kernels.cu.o
#ccache /usr/local/cuda/bin/nvcc -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_dattn_C \ 
#-D_dattn_C_EXPORTS -I/root/vllm/csrc -isystem /usr/include/python3.9 \
#-isystem /usr/local/lib/python3.9/dist-packages/torch/include \
#-isystem /usr/local/lib/python3.9/dist-packages/torch/include/torch/csrc/api/include \
#-isystem /usr/local/cuda/include -DONNX_NAMESPACE=onnx_c2 \
#--expt-relaxed-constexpr --expt-extended-lambda -O1 -std=c++17 \
#"--generate-code=arch=compute_80,code=[sm_80]" -Xcompiler=-fPIC \
#-DENABLE_FP8 -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/_dattn_C.dir/csrc/dattn/dattn.cu.o \
#-MF CMakeFiles/_dattn_C.dir/csrc/dattn/dattn.cu.o.d -x cu -c /root/vllm/csrc/dattn/dattn.cu \
#-o CMakeFiles/_dattn_C.dir/csrc/dattn/dattn.cu.o
#CMAKE_EXPORT_COMPILE_COMMANDS=ON pip install -e . --no-build-isolation -vvv
#cd ..
#export TARGET_MODULES="_C"
VERBOSE=1 pip install -e . --no-build-isolation -vvv
#pip install -e . -vvv
