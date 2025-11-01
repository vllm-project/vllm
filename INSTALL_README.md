# vLLM Installation Guide

Complete installation scripts for setting up vLLM with CPU KV Cache Offloading on Linux, macOS, and Windows.

## Features

- 🚀 **One-command installation** - Automated setup process
- 💾 **Model management** - Download models from HuggingFace or custom URLs
- 🔧 **Flexible configuration** - Choose installation directory, ports, and parameters
- 📊 **Optional web interface** - Monitor your vLLM server with the manager dashboard
- 🐧 **Multi-platform** - Supports Linux, macOS, and Windows
- 🎛️ **CPU Offloading** - Enable massive context windows with limited GPU RAM
- ⚙️ **Systemd/Windows Service** - Automatic startup and management

## Prerequisites

### Linux/macOS
- Python 3.8 or higher
- pip (Python package manager)
- git
- CUDA Toolkit (for GPU support)
- NVIDIA GPU with compatible drivers

### Windows
- Python 3.8 or higher
- pip (Python package manager)
- git
- CUDA Toolkit (for GPU support)
- NVIDIA GPU with compatible drivers
- PowerShell 5.1+ (included in Windows 10/11)
- Administrator privileges

## Quick Start

### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/datagram1/vllm.git
cd vllm

# Make the installer executable
chmod +x install.sh

# Run the installer
./install.sh
```

### Windows

```powershell
# Clone the repository
git clone https://github.com/datagram1/vllm.git
cd vllm

# Run PowerShell as Administrator and execute
Set-ExecutionPolicy Bypass -Scope Process -Force
.\install.ps1
```

## Installation Options

The installer will guide you through the following choices:

### 1. Installation Directory
- **Linux/macOS**:
  - Home directory: `~/.vllm`
  - System-wide: `/opt/vllm`
  - Custom path
- **Windows**:
  - User directory: `%USERPROFILE%\.vllm`
  - Program Files: `C:\Program Files\vLLM`
  - Custom path

### 2. Model Configuration
- Use existing model (provide local path)
- Download from HuggingFace (e.g., `Qwen/Qwen3-Coder-30B`)
- Download from custom URL

### 3. Server Settings
- Port (default: 8000)
- Maximum model length (default: 262144 for 256K context)
- GPU memory utilization (default: 0.88)
- CPU offload blocks (default: 16710)
- Tool parser (default: qwen3_coder)

### 4. Optional Components
- vLLM Manager web interface
- Manager port (default: 7999)

## Post-Installation

### Linux/macOS

```bash
# Check service status
sudo systemctl status vllm.service

# View logs
tail -f ~/.vllm/logs/vllm.log

# Test the API
curl http://localhost:8000/v1/models

# Access manager (if installed)
open http://localhost:7999
```

### Windows

```powershell
# Check service status
Get-Service vLLM

# View logs
Get-Content $HOME\.vllm\logs\vllm.log -Tail 50 -Wait

# Test the API
Invoke-WebRequest http://localhost:8000/v1/models

# Access manager (if installed)
Start-Process http://localhost:7999
```

## Service Management

### Linux/macOS

```bash
# Start service
sudo systemctl start vllm.service

# Stop service
sudo systemctl stop vllm.service

# Restart service
sudo systemctl restart vllm.service

# Enable auto-start on boot
sudo systemctl enable vllm.service

# View service logs
sudo journalctl -u vllm.service -f
```

### Windows

```powershell
# Start service
Start-Service vLLM

# Stop service
Stop-Service vLLM

# Restart service
Restart-Service vLLM

# Set auto-start (already configured by installer)
Set-Service vLLM -StartupType Automatic
```

## Configuration Files

After installation, you'll find:

```
Installation Directory/
├── bin/
│   ├── vllm_server.py      # Main server script
│   └── vllm_manager.py     # Manager web interface (optional)
├── models/                  # Downloaded models
├── logs/
│   ├── vllm.log            # Server logs
│   ├── vllm.err            # Server errors
│   ├── manager.log         # Manager logs (if installed)
│   └── manager.err         # Manager errors (if installed)
├── config/                  # Configuration files
└── venv/                    # Python virtual environment
```

## API Usage Examples

### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-path",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100
  }'
```

### Tool Calling (if tool parser enabled)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-path",
    "messages": [
      {"role": "user", "content": "What'\''s the weather in NYC?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }]
  }'
```

## CPU Offloading Explained

The CPU KV Cache Offloading feature allows you to:
- Run models with **256K context** using only **~24GB RAM** instead of 180GB
- Offload key-value cache to CPU memory
- Maintain high performance with automatic cache management

### How It Works

1. GPU handles model computations
2. KV cache is intelligently distributed between GPU and CPU
3. Active tokens stay in GPU for fast access
4. Inactive tokens moved to CPU to free GPU memory
5. Automatic prefetching minimizes latency

### Configuration

The installer automatically configures optimal settings:
- `num_cpu_blocks`: 16710 (provides ~24GB cache)
- `block_size`: 16 tokens per block
- `kv_connector`: OffloadingConnector
- `kv_role`: kv_both (handles both GPU and CPU)

## Troubleshooting

### Service won't start

**Linux/macOS:**
```bash
# Check logs for errors
sudo journalctl -u vllm.service -n 100 --no-pager

# Verify Python environment
source /path/to/vllm/venv/bin/activate
python -c "import vllm; print(vllm.__version__)"
```

**Windows:**
```powershell
# Check logs
Get-Content $HOME\.vllm\logs\vllm.err -Tail 50

# Verify service
Get-Service vLLM | Format-List *
```

### CUDA errors

- Ensure NVIDIA drivers are up to date
- Verify CUDA toolkit is installed
- Check `nvidia-smi` output
- Adjust `gpu-memory-utilization` to lower value (e.g., 0.7)

### Out of memory

- Reduce `num-cpu-blocks`
- Lower `max-model-len`
- Decrease `gpu-memory-utilization`
- Use smaller model

### Port already in use

```bash
# Find process using port 8000
# Linux/macOS
sudo lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Change port in configuration and restart service
```

## Uninstallation

### Linux/macOS

```bash
# Stop and disable services
sudo systemctl stop vllm.service
sudo systemctl disable vllm.service
sudo systemctl stop vllm-manager.service
sudo systemctl disable vllm-manager.service

# Remove service files
sudo rm /etc/systemd/system/vllm.service
sudo rm /etc/systemd/system/vllm-manager.service
sudo systemctl daemon-reload

# Remove installation directory
rm -rf ~/.vllm  # or /opt/vllm
```

### Windows

```powershell
# Stop and remove service
Stop-Service vLLM
& "$HOME\.vllm\bin\nssm.exe" remove vLLM confirm

# Remove installation directory
Remove-Item -Recurse -Force $HOME\.vllm
```

## Advanced Configuration

### Custom Model Configuration

Edit `{install_dir}/bin/vllm_server.py` to modify:
- Model path
- Context length
- GPU memory allocation
- Tool parser settings
- Other vLLM parameters

After changes, restart the service.

### Environment Variables

**Linux/macOS** (`/etc/systemd/system/vllm.service`):
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="VLLM_WORKER_MULTIPROC_METHOD=spawn"
```

**Windows** (via NSSM):
```powershell
nssm set vLLM AppEnvironmentExtra CUDA_VISIBLE_DEVICES=0,1
```

## Support

- **Issues**: https://github.com/datagram1/vllm/issues
- **Discussions**: https://github.com/datagram1/vllm/discussions
- **Documentation**: https://github.com/datagram1/vllm

## License

Apache 2.0 - See LICENSE file for details

## Credits

- Original vLLM project: https://github.com/vllm-project/vllm
- CPU KV Cache Offloading implementation
- Enhanced for massive context windows (256K+)
