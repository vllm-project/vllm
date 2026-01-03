#!/bin/bash
set -e

# Configuration
MAMBA_VERSION="v2.2.6.post3"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
DEFAULT_HIP_ARCH="gfx942"

# Logging helpers
log_info()    { echo "[INFO]: $*" >&2; }
log_success() { echo "[SUCCESS]: $*" >&2; }
log_warning() { echo "[WARNING]: $*" >&2; }
log_error()   { echo "[ERROR]: $*" >&2; }
log_step()    { echo "" >&2; echo "=== $* ===" >&2; }

# Detect GPU architecture from rocminfo
detect_hip_arch() {
    if ! command -v rocminfo &> /dev/null; then
        log_warning "rocminfo not found, cannot auto-detect GPU architecture"
        return 1
    fi
    
    local detected
    detected=$(rocminfo 2>/dev/null | grep -oP 'Name:\s+\Kgfx[0-9a-z]+' | sort -u | tr '\n' ';' | sed 's/;$//')
    
    if [[ -z "$detected" ]]; then
        log_warning "rocminfo did not return any GPU architectures"
        return 1
    fi
    
    echo "$detected"
}

# Get HIP architecture
get_hip_arch() {
    if [[ -n "${HIP_ARCHITECTURES:-}" ]]; then
        log_info "Using user-specified HIP_ARCHITECTURES: $HIP_ARCHITECTURES"
        echo "$HIP_ARCHITECTURES"
        return 0
    fi
    
    log_info "Attempting to auto-detect GPU architecture..."
    local detected
    if detected=$(detect_hip_arch); then
        log_success "Auto-detected GPU architecture: $detected"
        echo "$detected"
        return 0
    fi
    
    log_warning "Could not auto-detect GPU architecture, falling back to default: $DEFAULT_HIP_ARCH"
    echo "$DEFAULT_HIP_ARCH"
}

# Detect ROCm version
get_rocm_version() {
    local version_file="$ROCM_PATH/.info/version"
    
    if [[ -f "$version_file" ]]; then
        cut -d'-' -f1 < "$version_file"
    else
        log_warning "Could not detect ROCm version (missing $version_file)"
        echo "unknown"
    fi
}

# Apply ROCm 6.0 bf16 patch if needed
apply_rocm60_patch() {
    local patch_file="$1"
    local target_file="$ROCM_PATH/include/hip/amd_detail/amd_hip_bf16.h"
    
    if [[ ! -f "$target_file" ]]; then
        log_warning "Patch target not found: $target_file (skipping)"
        return 0
    fi
    
    if [[ ! -f "$patch_file" ]]; then
        log_warning "Patch file not found: $patch_file (skipping)"
        return 0
    fi
    
    log_info "Applying ROCm 6.0 bf16 patch to $target_file"
    if [[ -w "$target_file" ]]; then
        patch --forward "$target_file" < "$patch_file" || true
    else
        log_info "Requesting sudo for patching system file..."
        sudo patch --forward "$target_file" < "$patch_file" || true
    fi
    log_success "Patch applied"
}

# Fix ROCm 7.0 warp_size constexpr issue
# The original CUDA file is reverse_scan.cuh - hipify converts it to reverse_scan_hip.cuh
apply_rocm70_warp_size_fix() {
    local source_dir="$1"
    local file="$source_dir/csrc/selective_scan/reverse_scan.cuh"
    
    if [[ ! -f "$file" ]]; then
        log_error "reverse_scan.cuh not found at: $file"
        log_info "Available files in csrc/selective_scan:"
        ls -la "$source_dir/csrc/selective_scan/" >&2 || true
        return 1
    fi
    
    log_info "Patching $file for ROCm 7.0 compatibility..."
    
    # The problem: In ROCm 7.0, HIPCUB_WARP_THREADS expands to rocprim::warp_size()
    # which is NOT constexpr, but the code uses it in a constexpr context.
    #
    # Original code around line 106-108:
    #     #ifdef __HIP_PLATFORM_AMD__
    #         #define WARP_THREADS HIPCUB_WARP_THREADS
    #     #else
    #         #define WARP_THREADS CUB_PTX_WARP_THREADS
    #     #endif
    #     static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == WARP_THREADS);
    #
    # Fix: Replace HIPCUB_WARP_THREADS with hardcoded 64 (AMD wavefront size)
    
    # Check if already patched
    if grep -q "WARP_THREADS 64" "$file"; then
        log_info "File already patched, skipping"
        return 0
    fi
    
    # Replace the HIPCUB_WARP_THREADS with 64
    sed -i 's/#define WARP_THREADS HIPCUB_WARP_THREADS/#define WARP_THREADS 64  \/\/ Hardcoded for ROCm 7.0+ (rocprim::warp_size() not constexpr)/g' "$file"
    
    # Verify the fix was applied
    if grep -q "WARP_THREADS 64" "$file"; then
        log_success "ROCm 7.0 warp_size fix applied to reverse_scan.cuh"
    else
        log_error "Failed to apply ROCm 7.0 fix to reverse_scan.cuh"
        return 1
    fi
}

# Main
main() {
    log_step "Mamba ROCm Installation"
    
    HIP_ARCH=$(get_hip_arch)
    ROCM_VERSION=$(get_rocm_version)
    
    log_info "Mamba version:    $MAMBA_VERSION"
    log_info "HIP architecture: $HIP_ARCH"
    log_info "ROCm version:     $ROCM_VERSION"
    log_info "ROCm path:        $ROCM_PATH"
    
    WORK_DIR=$(mktemp -d)
    trap "rm -rf $WORK_DIR" EXIT
    
    log_step "Cloning Repository"
    log_info "Cloning mamba $MAMBA_VERSION into $WORK_DIR"
    git clone --quiet --depth 1 --branch "$MAMBA_VERSION" https://github.com/state-spaces/mamba.git "$WORK_DIR/mamba_ssm"
    cd "$WORK_DIR/mamba_ssm"
    log_success "Repository cloned"
    
    # Apply ROCm 6.0 patch if needed
    if [[ "$ROCM_VERSION" == 6.0* ]]; then
        log_step "Applying ROCm 6.0 Patch"
        apply_rocm60_patch "rocm_patch/rocm6_0.patch"
    fi
    
    # Apply ROCm 7.0 warp_size fix
    if [[ "$ROCM_VERSION" == 7.* ]]; then
        log_step "Applying ROCm 7.0 Compatibility Fix"
        apply_rocm70_warp_size_fix "$WORK_DIR/mamba_ssm"
    fi
    
    log_step "Building and Installing"
    export HIP_ARCHITECTURES="$HIP_ARCH"
    export MAMBA_FORCE_BUILD="TRUE"
    
    log_info "Building for architecture: $HIP_ARCH"
    log_info "This may take several minutes..."
    pip install --no-cache-dir --no-build-isolation --verbose .
    
    log_step "Verifying Installation"
    if python -c "import mamba_ssm; print(f'[SUCCESS]: mamba_ssm {mamba_ssm.__version__} installed successfully')"; then
        log_success "Installation complete! mamba-ssm is ready to use."
    else
        log_error "Installation verification failed"
        exit 1
    fi
}

main "$@"