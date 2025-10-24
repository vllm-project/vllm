# Guideline

## Set Env

1. Docker image:

   ```shell
   rocm/vllm-private:355_wip_2b4cb8a11_1021
   ```

2. Install aiter dev/perf branch:

   ```shell
   pip uninstall aiter
   git clone -b dev/perf git@github.com:ROCm/aiter.git
   cd aiter
   git submodule sync && git submodule update --init --recursive
   python3 setup.py install
   ```

3. Install rocm/vLLM dev/perf branch:

   ```shell
   pip uninstall vllm
   git clone -b dev/perf git@github.com:ROCm/vllm.git
   cd vllm
   python3 -m pip install -r requirements/common.txt
   export PYTORCH_ROCM_ARCH="gfx950"
   python3 setup.py develop
   ```
