export PYTHON_ENV_DIR="${TT_METAL_HOME}/build/python_env_vllm"
export VLLM_TARGET_DEVICE="tt"

# to create vllm env (first time):
# 1. setup tt-metal env vars 
# 2. source $vllm_dir/tt_metal/setup-metal.sh (this script)
# 3. build and create tt-metal env as usual
# 4. source $PYTHON_ENV_DIR/bin/activate 
# 5. pip3 install --upgrade pip
# 6. cd $vllm_dir && pip install -e .

# to activate (after first time):
# 1. source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
