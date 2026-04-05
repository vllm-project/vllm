# RoCM Expert Parallel(EP) Kernel installation

This directory contains installation scripts to install all2all EP kernels used on RoCM. At the time of writing it
installs:

- MoRI

Note: As a side-effect it also installs Aiter, so MoRI could be used with AiterExperts.

## Usage

### Install kernels with automatic GPU_ARCHS detection

```bash
bash install_python_libraries.sh
```

### Install kernels for specific GPU_ARCHS

```bash
GPU_ARCHS="gfx942;gfx950" bash install_python_libraries.sh
```

The script has hard-coded repository and hash values picked from Dockerfile.rocm_base. The script avoids installation
if it finds an existing installation of the package. To force a reinstall, use `--force-install`.
