# vLLM (sy)

Quick setup and run guide for the vLLM repo.

## Install

```bash
git clone {our repo .git}
cd vllm
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
# Don't forget to do "module load cuda/12.8 && module load gcc" when using tacc
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
grep -v '^torch==' requirements/build.txt | uv pip install -r -
uv pip install -e . --no-build-isolation
```

Optional (if you use a shared cache):

```bash
# ln -s $WORK/.cache ~/.cache
```

## Run example

```bash
uv run bash examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh
```

## Troubleshooting

- If the encoder or PD worker fails, check `./logs`.
- Example run output: `examples/online_serving/disaggregated_encoder/1e1pd.txt`
