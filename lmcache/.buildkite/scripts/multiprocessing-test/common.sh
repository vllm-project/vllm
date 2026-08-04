#!/bin/bash
# Common utilities for multiprocessing tests
# Source this file to use shared functions and variables

# Get the directory where this script is located
COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$COMMON_SCRIPT_DIR/../../../.." && pwd)"
LMCACHE_DIR="$(cd "$COMMON_SCRIPT_DIR/../../.." && pwd)"

# Common configuration
BUILD_ID="${BUILD_ID:-local_$(date +%Y%m%d_%H%M%S)}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"

# Setup virtual environment
# Usage: setup_venv [packages...]
# Example: setup_venv openai pandas matplotlib
setup_venv() {
    local packages=("$@")
    
    echo "=== Setting up virtual environment ==="
    
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating virtual environment with uv..."
        
        # Check if uv is available
        if ! command -v uv &> /dev/null; then
            echo "uv not found, installing..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
        
        # Create venv with uv
        uv venv "$VENV_DIR"
        echo "Virtual environment created"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated"
    
    # Install dependencies if provided
    if [ ${#packages[@]} -gt 0 ]; then
        echo "Installing dependencies: ${packages[*]}"
        uv pip install "${packages[@]}" --quiet
        echo "Dependencies installed"
    fi
    
    echo ""
}

# Ensure results directory exists
ensure_results_dir() {
    mkdir -p "$RESULTS_DIR"
    echo "Results directory: $RESULTS_DIR"
}

