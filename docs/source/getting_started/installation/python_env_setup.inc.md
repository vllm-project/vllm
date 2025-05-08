You can create a new Python environment using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html):

```console
# (Recommended) Create a new conda environment.
conda create -n vllm python=3.12 -y
conda activate vllm
```

:::{note}
[PyTorch has deprecated the conda release channel](https://github.com/pytorch/pytorch/issues/138506). If you use `conda`, please only use it to create Python environment rather than installing packages.
:::

Or you can create a new Python environment using [uv](https://docs.astral.sh/uv/), a very fast Python environment manager. Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, you can create a new Python environment using the following command:

```console
# (Recommended) Create a new uv environment. Use `--seed` to install `pip` and `setuptools` in the environment.
uv venv --python 3.12 --seed
source .venv/bin/activate
```
