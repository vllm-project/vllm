const { spawn } = require('child_process');
const path = require('path');

const python = path.join(__dirname, '.venv', 'bin', 'python');
const port = '52099';

const args = [
  '-m', 'vllm.entrypoints.openai.api_server',
  '--model', '/data1/meituan-longcat/LongCat-Next',
  '--tensor-parallel-size', '8',
  '--data-parallel-size', '1',
  '--enable-expert-parallel',
  '--enable-ep-weight-filter',
  '--pipeline-parallel-size', '1',
  '--dtype', 'bfloat16',
  '--trust-remote-code',
  '--port', port,
  '--max-model-len', '8192',
  '--gpu-memory-utilization', '0.90',
  '--mm-encoder-tp-mode', 'data',
  '--disable-custom-all-reduce',
  '--max-num-seqs', '1',
  '--max-num-batched-tokens', '512',
  '--enforce-eager',
];

const proc = spawn(python, args, {
  cwd: __dirname,
  stdio: 'inherit',
  env: {
    ...process.env,
    'NCCL_P2P_DISABLE': '1',
    'VLLM_NCCL_P2P_CGA_SIZE': '0',
    'VLLM_LOGGING_LEVEL': 'DEBUG',
    'CUDA_LAUNCH_BLOCKING': '1',
    'NCCL_DEBUG': 'INFO',
  },
});

proc.on('close', (code) => {
  process.exit(code ?? 0);
});
