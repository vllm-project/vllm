First, install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```console
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

Second, install Python packages for vLLM CPU backend building:

```console
pip install --upgrade pip
pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Finally, build and install vLLM CPU backend:

```console
VLLM_TARGET_DEVICE=cpu python setup.py install
```
