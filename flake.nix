{
  description = "vLLM Development Environment with CUDA support";

  inputs = {
    # Use stable 25.11 to avoid glibc 2.42 + CUDA 12.8 incompatibility
    # See: https://github.com/NixOS/nixpkgs/pull/484031
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # CUDA packages
        cudaPackages = pkgs.cudaPackages_12_8;

        fhsEnv = pkgs.buildFHSEnv {
          name = "vllm-dev-env";

          targetPkgs = pkgs':
            with pkgs'; [
              # Python
              python312
              python312Packages.pip
              python312Packages.setuptools
              python312Packages.wheel
              uv

              # CUDA Toolkit
              cudaPackages.cudatoolkit
              cudaPackages.cuda_nvcc
              cudaPackages.cuda_cudart
              cudaPackages.cuda_cupti
              cudaPackages.cuda_nvrtc
              cudaPackages.libcublas
              cudaPackages.libcufft
              cudaPackages.libcurand
              cudaPackages.libcusolver
              cudaPackages.libcusparse
              cudaPackages.cudnn
              cudaPackages.nccl

              # Build tools
              cmake
              ninja
              gcc13
              gnumake
              pkg-config

              # System libraries
              zlib
              stdenv.cc.cc.lib
              linuxPackages.nvidia_x11
              libGL
              libGLU
              xorg.libX11
              xorg.libXext
              xorg.libXrender
              xorg.libXi

              # Development tools
              git
              git-lfs
              curl
              wget
              unzip
              jq
              tree

              # Editors
              zed-editor

              # Shells
              zsh
              bash

              # Linting
              ruff
              pre-commit
            ];

          profile = ''
            echo "🚀 vLLM Development Environment"
            echo "================================"

            # Set CUDA environment variables
            export CUDA_HOME="${cudaPackages.cudatoolkit}"
            export CUDA_PATH="$CUDA_HOME"
            export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"

            # Add CUDA to paths
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudnn}/lib:${cudaPackages.nccl}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
            export LIBRARY_PATH="$LD_LIBRARY_PATH"

            # CUDA include paths for cmake
            export CPATH="${cudaPackages.cudatoolkit}/include:${cudaPackages.cudnn}/include:$CPATH"
            export CMAKE_CUDA_COMPILER="${cudaPackages.cuda_nvcc}/bin/nvcc"

            # For torch/cmake to find CUDA
            export CUDA_INCLUDE_DIRS="${cudaPackages.cudatoolkit}/include"
            export CUDA_LIBRARIES="${cudaPackages.cudatoolkit}/lib"

            # vLLM specific
            export VLLM_TARGET_DEVICE="cuda"
            export MAX_JOBS=20  # Optimized for 62GB RAM (actual usage ~1-2GB per job)

            # Create and activate uv virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "📦 Creating Python virtual environment..."
              uv venv --python python3.12 --prompt "vllm-dev"
            fi

            # Activate the virtual environment
            source .venv/bin/activate
            export VIRTUAL_ENV_PROMPT="vllm-dev"

            echo ""
            echo "✅ Python:    $(python --version)"
            echo "✅ uv:        $(uv --version)"
            echo "✅ CUDA:      $(nvcc --version 2>/dev/null | grep release | awk '{print $6}')"
            echo "✅ CUDA_HOME: $CUDA_HOME"
            echo "✅ CMAKE:     $(cmake --version | head -1)"
          '';

          runScript = ''
            # Set shell for the environment
            SHELL=${pkgs.zsh}/bin/zsh

            export SSL_CERT_FILE="/etc/ssl/certs/ca-bundle.crt"

            echo ""
            echo "📚 vLLM Development Quick Reference:"
            echo ""
            echo "🔧 Setup (first time):"
            echo "  uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128"
            echo "  uv pip install -e .                    # Editable install (builds CUDA kernels)"
            echo ""
            echo "🧪 Testing:"
            echo "  python -c 'import vllm; print(vllm.__version__)'"
            echo "  python .agent/handoff/test_deepseek_ocr2.py"
            echo ""
            echo "🏃 Running:"
            echo "  vllm serve <model>                     # Start API server"
            echo "  python -m vllm.entrypoints.openai.api_server --model <model>"
            echo ""
            echo "🔨 Building:"
            echo "  uv pip install -e . --no-build-isolation  # Rebuild after changes"
            echo ""
            echo "🚀 Ready to develop vLLM!"
            echo ""

            # Start zsh shell
            exec ${pkgs.zsh}/bin/zsh
          '';
        };
      in {
        devShells.default = pkgs.mkShell {
          shellHook = ''
            exec ${fhsEnv}/bin/vllm-dev-env
          '';
        };

        packages.default = pkgs.python312;
      }
    );
}
