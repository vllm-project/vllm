# Lambda Labs Instance Setup

## Instance Details
- **Instance ID**: 0b84a041d4544e72ad453da7bf2c5b38
- **IP Address**: 132.145.142.82
- **Type**: gpu_1x_a100_sxm4 (1x A100 40GB)
- **Region**: us-east-1 (Virginia, USA)
- **Cost**: $1.29/hour
- **SSH Key**: sheikh

## Hardware Specs
- **GPU**: NVIDIA A100-SXM4-40GB
- **CUDA**: 12.8
- **Driver**: 570.148.08
- **CPU**: 30 vCPUs
- **RAM**: 200 GiB
- **Storage**: 512 GiB
- **Python**: 3.10.12

## Connection
```bash
ssh ubuntu@132.145.142.82
```

## vLLM Setup
- **Repository**: https://github.com/sheikheddy/vllm.git
- **Location**: ~/vllm
- **Branch**: main (with INT4 + LoRA support)
- **Installation**: In progress (compiling CUDA kernels)

## Helper Script
Use the `lambda_instance.sh` script in this directory:

```bash
# Check instance status
./lambda_instance.sh status

# Get IP address
./lambda_instance.sh ip

# SSH into instance
./lambda_instance.sh ssh

# Terminate instance when done
./lambda_instance.sh terminate
```

## Important Notes
- Remember to terminate the instance when done to avoid charges
- The instance costs $1.29/hour
- vLLM is being installed in editable mode for development
- Jupyter Lab is pre-installed and running (token: 4e1bcc82a5cc4c7d905fe893a3578604)

## Next Steps
Once vLLM installation completes:
1. Test the installation: `python3 -c "import vllm; print(vllm.__version__)"`
2. Run your INT4 LoRA tests
3. Verify GPU availability: `nvidia-smi`
