#!/usr/bin/env bash
set -e

# vLLM with CPU Offloading - Installation Script
# Supports Ubuntu/Debian and RHEL/CentOS/Fedora

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INSTALL_DIR=""
MODEL_PATH=""
MODEL_URL=""
INSTALL_MANAGER="no"
VLLM_PORT=8000
MANAGER_PORT=7999
MAX_MODEL_LEN=262144
GPU_MEMORY_UTIL=0.88
NUM_CPU_BLOCKS=16710
TOOL_PARSER="qwen3_coder"

# Print functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running as root (we don't want that for most operations)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root directly."
        print_info "It will ask for sudo password when needed."
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        print_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi

    print_info "Detected OS: $OS $OS_VERSION"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    local missing_deps=()

    # Check Python 3.8+
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    else
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        missing_deps+=("python3-pip")
    else
        print_success "pip found"
    fi

    # Check git
    if ! command -v git &> /dev/null; then
        missing_deps+=("git")
    else
        print_success "git found"
    fi

    # Check curl
    if ! command -v curl &> /dev/null; then
        missing_deps+=("curl")
    else
        print_success "curl found"
    fi

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "nvidia-smi not found. GPU support may not be available."
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        echo ""
        echo "Install them with:"
        if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
            echo "  sudo apt-get update && sudo apt-get install -y ${missing_deps[*]}"
        elif [[ "$OS" == "rhel" ]] || [[ "$OS" == "centos" ]] || [[ "$OS" == "fedora" ]]; then
            echo "  sudo dnf install -y ${missing_deps[*]}"
        fi
        exit 1
    fi
}

# Interactive configuration
configure_installation() {
    print_header "Installation Configuration"

    # Choose installation directory
    echo ""
    echo "Where would you like to install vLLM?"
    echo "1) Home directory (~/.vllm)"
    echo "2) /opt/vllm (requires sudo)"
    echo "3) Custom path"
    read -p "Choice [1]: " install_choice
    install_choice=${install_choice:-1}

    case $install_choice in
        1)
            INSTALL_DIR="$HOME/.vllm"
            ;;
        2)
            INSTALL_DIR="/opt/vllm"
            ;;
        3)
            read -p "Enter custom path: " INSTALL_DIR
            INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac

    print_info "Installation directory: $INSTALL_DIR"

    # Model configuration
    echo ""
    echo "Model Configuration:"
    echo "1) Use existing model (provide path)"
    echo "2) Download model from HuggingFace"
    echo "3) Download model from custom URL"
    read -p "Choice [1]: " model_choice
    model_choice=${model_choice:-1}

    case $model_choice in
        1)
            read -p "Enter model path: " MODEL_PATH
            MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
            if [[ ! -d "$MODEL_PATH" ]]; then
                print_error "Model path does not exist: $MODEL_PATH"
                exit 1
            fi
            ;;
        2)
            read -p "Enter HuggingFace model ID (e.g., Qwen/Qwen3-Coder-30B): " hf_model_id
            MODEL_PATH="$INSTALL_DIR/models/$(basename $hf_model_id)"
            MODEL_URL="https://huggingface.co/$hf_model_id"
            ;;
        3)
            read -p "Enter model download URL: " MODEL_URL
            read -p "Enter local model name: " model_name
            MODEL_PATH="$INSTALL_DIR/models/$model_name"
            ;;
    esac

    # Server configuration
    echo ""
    read -p "vLLM server port [8000]: " port_input
    VLLM_PORT=${port_input:-8000}

    read -p "Maximum model length [262144]: " max_len_input
    MAX_MODEL_LEN=${max_len_input:-262144}

    read -p "GPU memory utilization (0.0-1.0) [0.88]: " gpu_mem_input
    GPU_MEMORY_UTIL=${gpu_mem_input:-0.88}

    read -p "Number of CPU blocks for offloading [16710]: " cpu_blocks_input
    NUM_CPU_BLOCKS=${cpu_blocks_input:-16710}

    read -p "Tool parser (leave empty for none) [qwen3_coder]: " tool_parser_input
    TOOL_PARSER=${tool_parser_input:-qwen3_coder}

    # Manager installation
    echo ""
    read -p "Install vLLM Manager web interface? (yes/no) [no]: " install_mgr
    INSTALL_MANAGER=${install_mgr:-no}

    if [[ "$INSTALL_MANAGER" == "yes" ]]; then
        read -p "Manager port [7999]: " mgr_port_input
        MANAGER_PORT=${mgr_port_input:-7999}
    fi

    # Summary
    echo ""
    print_header "Installation Summary"
    echo "Installation directory: $INSTALL_DIR"
    echo "Model path: $MODEL_PATH"
    [[ -n "$MODEL_URL" ]] && echo "Model URL: $MODEL_URL"
    echo "vLLM port: $VLLM_PORT"
    echo "Max model length: $MAX_MODEL_LEN"
    echo "GPU memory utilization: $GPU_MEMORY_UTIL"
    echo "CPU blocks: $NUM_CPU_BLOCKS"
    echo "Tool parser: ${TOOL_PARSER:-none}"
    echo "Install manager: $INSTALL_MANAGER"
    [[ "$INSTALL_MANAGER" == "yes" ]] && echo "Manager port: $MANAGER_PORT"
    echo ""
    read -p "Continue with installation? (yes/no) [yes]: " confirm
    confirm=${confirm:-yes}

    if [[ "$confirm" != "yes" ]]; then
        print_info "Installation cancelled"
        exit 0
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"

    if [[ "$INSTALL_DIR" == /opt/* ]]; then
        sudo mkdir -p "$INSTALL_DIR"/{bin,models,logs,config}
        sudo chown -R $USER:$USER "$INSTALL_DIR"
    else
        mkdir -p "$INSTALL_DIR"/{bin,models,logs,config}
    fi

    print_success "Directories created"
}

# Install Python dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"

    # Create virtual environment
    if [[ ! -d "$INSTALL_DIR/venv" ]]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$INSTALL_DIR/venv"
    fi

    # Activate venv
    source "$INSTALL_DIR/venv/bin/activate"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    # Install vLLM
    print_info "Installing vLLM (this may take a while)..."
    pip install vllm==0.11.0

    # Install additional dependencies
    print_info "Installing additional dependencies..."
    pip install nvidia-ml-py requests aiohttp fastapi uvicorn

    print_success "Dependencies installed"
}

# Download model if needed
download_model() {
    if [[ -z "$MODEL_URL" ]]; then
        return
    fi

    print_header "Downloading Model"

    mkdir -p "$MODEL_PATH"

    if [[ "$MODEL_URL" == *"huggingface.co"* ]]; then
        print_info "Downloading from HuggingFace..."
        pip install huggingface_hub
        python3 << EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_URL##*/}",
    local_dir="$MODEL_PATH",
    resume_download=True
)
EOF
    else
        print_info "Downloading from URL..."
        curl -L "$MODEL_URL" -o "$MODEL_PATH/model.tar.gz"
        tar -xzf "$MODEL_PATH/model.tar.gz" -C "$MODEL_PATH"
        rm "$MODEL_PATH/model.tar.gz"
    fi

    print_success "Model downloaded"
}

# Create server script
create_server_script() {
    print_header "Creating Server Script"

    cat > "$INSTALL_DIR/bin/vllm_server.py" << EOF
#!/usr/bin/env python3
"""
vLLM OpenAI API Server with CPU KV Cache Offloading
Generated by vLLM Installer v${VERSION}
"""

import sys
import runpy
from vllm.config import KVTransferConfig

# Configure CPU offloading for KV cache
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "num_cpu_blocks": ${NUM_CPU_BLOCKS},
        "block_size": 16,
    },
)

# Build command line arguments
sys.argv = [
    "vllm.entrypoints.openai.api_server",
    "--model", "${MODEL_PATH}",
    "--dtype", "auto",
    "--max-model-len", "${MAX_MODEL_LEN}",
    "--gpu-memory-utilization", "${GPU_MEMORY_UTIL}",
    "--enforce-eager",
    "--max-num-seqs", "8",
    "--tensor-parallel-size", "1",
    "--enable-prefix-caching",
    "--host", "0.0.0.0",
    "--port", "${VLLM_PORT}",
EOF

    if [[ -n "$TOOL_PARSER" ]]; then
        cat >> "$INSTALL_DIR/bin/vllm_server.py" << EOF
    "--tool-call-parser", "${TOOL_PARSER}",
    "--enable-auto-tool-choice",
EOF
    fi

    cat >> "$INSTALL_DIR/bin/vllm_server.py" << 'EOF'
]

# Monkey-patch the KVTransferConfig
import vllm.engine.arg_utils
original_create_engine_config = vllm.engine.arg_utils.EngineArgs.create_engine_config

def patched_create_engine_config(self, *args, **kwargs):
    if self.kv_transfer_config is None:
        self.kv_transfer_config = kv_transfer_config
    return original_create_engine_config(self, *args, **kwargs)

vllm.engine.arg_utils.EngineArgs.create_engine_config = patched_create_engine_config

# Run the API server
if __name__ == "__main__":
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
EOF

    chmod +x "$INSTALL_DIR/bin/vllm_server.py"
    print_success "Server script created"
}

# Create systemd service
create_systemd_service() {
    print_header "Creating systemd Service"

    sudo tee /etc/systemd/system/vllm.service > /dev/null << EOF
[Unit]
Description=vLLM Large Language Model Server with CPU Offloading
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/bin/vllm_server.py
StandardOutput=append:$INSTALL_DIR/logs/vllm.log
StandardError=append:$INSTALL_DIR/logs/vllm.err
Restart=on-failure
RestartSec=10
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable vllm.service

    print_success "systemd service created and enabled"
}

# Install manager
install_manager() {
    if [[ "$INSTALL_MANAGER" != "yes" ]]; then
        return
    fi

    print_header "Installing vLLM Manager"

    # Copy manager script
    if [[ -f "$SCRIPT_DIR/tools/vllm_manager.py" ]]; then
        cp "$SCRIPT_DIR/tools/vllm_manager.py" "$INSTALL_DIR/bin/"
        chmod +x "$INSTALL_DIR/bin/vllm_manager.py"
    else
        print_warning "Manager script not found in tools/vllm_manager.py"
        print_warning "Skipping manager installation"
        return
    fi

    # Create manager service
    sudo tee /etc/systemd/system/vllm-manager.service > /dev/null << EOF
[Unit]
Description=vLLM Manager Web Interface
After=network.target vllm.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="VLLM_PORT=${VLLM_PORT}"
Environment="MANAGER_PORT=${MANAGER_PORT}"
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/bin/vllm_manager.py
StandardOutput=append:$INSTALL_DIR/logs/manager.log
StandardError=append:$INSTALL_DIR/logs/manager.err
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable vllm-manager.service

    print_success "Manager installed and enabled"
}

# Apply CPU offload fix
apply_cpu_offload_fix() {
    print_header "Applying CPU Offload Memory Fix"

    local kv_cache_utils="$INSTALL_DIR/venv/lib/python*/site-packages/vllm/v1/core/kv_cache_utils.py"

    if [[ -f "$SCRIPT_DIR/vllm/v1/core/kv_cache_utils.py" ]]; then
        cp "$SCRIPT_DIR/vllm/v1/core/kv_cache_utils.py" $kv_cache_utils
        print_success "CPU offload fix applied"
    else
        print_warning "kv_cache_utils.py fix not found in source"
        print_warning "You may need to apply the fix manually"
    fi
}

# Start services
start_services() {
    print_header "Starting Services"

    print_info "Starting vLLM service..."
    sudo systemctl start vllm.service

    sleep 5

    if sudo systemctl is-active --quiet vllm.service; then
        print_success "vLLM service started"
    else
        print_error "Failed to start vLLM service"
        print_info "Check logs: sudo journalctl -u vllm.service -f"
        exit 1
    fi

    if [[ "$INSTALL_MANAGER" == "yes" ]]; then
        print_info "Starting manager service..."
        sudo systemctl start vllm-manager.service

        if sudo systemctl is-active --quiet vllm-manager.service; then
            print_success "Manager service started"
        else
            print_warning "Failed to start manager service"
        fi
    fi
}

# Print final instructions
print_instructions() {
    print_header "Installation Complete!"

    echo ""
    echo -e "${GREEN}vLLM is now installed and running!${NC}"
    echo ""
    echo "Service Management:"
    echo "  Start:   sudo systemctl start vllm.service"
    echo "  Stop:    sudo systemctl stop vllm.service"
    echo "  Restart: sudo systemctl restart vllm.service"
    echo "  Status:  sudo systemctl status vllm.service"
    echo "  Logs:    tail -f $INSTALL_DIR/logs/vllm.log"
    echo ""
    echo "API Endpoint:"
    echo "  http://localhost:${VLLM_PORT}/v1/chat/completions"
    echo ""
    echo "Test the server:"
    echo "  curl http://localhost:${VLLM_PORT}/v1/models"
    echo ""

    if [[ "$INSTALL_MANAGER" == "yes" ]]; then
        echo "Manager Interface:"
        echo "  http://localhost:${MANAGER_PORT}"
        echo "  Service: sudo systemctl status vllm-manager.service"
        echo ""
    fi

    echo "Configuration:"
    echo "  Installation: $INSTALL_DIR"
    echo "  Model: $MODEL_PATH"
    echo "  Logs: $INSTALL_DIR/logs/"
    echo ""
    echo -e "${BLUE}For more information, visit: https://github.com/datagram1/vllm${NC}"
}

# Main installation flow
main() {
    print_header "vLLM with CPU Offloading Installer v${VERSION}"

    check_root
    detect_os
    check_prerequisites
    configure_installation
    create_directories
    install_dependencies
    download_model
    create_server_script
    apply_cpu_offload_fix
    create_systemd_service
    install_manager
    start_services
    print_instructions
}

# Run main
main "$@"
