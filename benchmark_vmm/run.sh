cmake /home/cpchung/dev/vllm -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/tmp/tmpfdoj72eo.build-lib/vllm -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=/tmp/tmp9uni28nq.build-temp -DVLLM_TARGET_DEVICE=cuda -DVLLM_PYTHON_EXECUTABLE=/home/cpchung/miniconda3/bin/python -DNVCC_THREADS=1 -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile=6


cmake --build . -j=6 --target=_moe_C --target=_vmm_C --target=_C