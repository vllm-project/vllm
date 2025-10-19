# Verification Guide: Go Containerregistry Integration

This guide helps verify that the go-containerregistry integration is working correctly.

## Quick Verification Steps

### 1. Build the Go Library

```bash
cd vllm/model_executor/model_loader/oci_go
./build.sh
```

**Expected output:**
```
Building OCI Go client library...
Downloading Go dependencies...
Building for Linux...
Built liboci.so
Building static archive...
Built liboci.a
Build complete!
```

**Verify files created:**
```bash
ls -lh liboci.*
# Should show:
# liboci.a   (static archive, ~25 MB)
# liboci.h   (C header file)
# liboci.so  (shared library, ~10 MB) [Linux]
# or liboci.dylib (shared library) [macOS]
```

### 2. Run Standalone Tests

```bash
cd vllm/model_executor/model_loader/oci_go
python3 test_client.py
```

**Expected output:**
```
======================================================================
OCI Go Client Library Tests
======================================================================
Testing library loading...
✓ Library loaded successfully from .../liboci.so

Testing function signatures...
✓ Function signatures set up successfully

Testing public image manifest pull...
This test requires network access
Pulling manifest for: docker.io/library/alpine:latest
✓ Successfully pulled manifest
  Schema version: 2
  Media type: application/vnd.oci.image.index.v1+json
  Config digest: N/A...
  Number of layers: 0

======================================================================
Results: 3 passed, 0 failed
======================================================================
```

### 3. Test Authentication (Optional)

If you have a private registry:

```bash
# Authenticate
docker login ghcr.io
# Enter your username and token

# Test with a private image
python3 << 'EOF'
import sys
sys.path.insert(0, '/path/to/vllm')
from vllm.model_executor.model_loader.oci_go_client import OciGoClient

client = OciGoClient()
error = client.test_authentication("ghcr.io/your-org/your-image:tag")
if error:
    print(f"Authentication failed: {error}")
else:
    print("✓ Authentication successful")
EOF
```

### 4. Verify Python Wrapper

```bash
cd /path/to/vllm
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# Test import (may fail if torch not installed, but that's OK)
try:
    from vllm.model_executor.model_loader.oci_go_client import OciGoClient
    print("✓ Python wrapper imports successfully")
    
    # Test initialization
    try:
        client = OciGoClient()
        print("✓ Go client initializes successfully")
        print(f"  Library path: {client.lib._name}")
    except RuntimeError as e:
        if "OCI Go library not found" in str(e):
            print("✗ Go library not built. Run: cd vllm/model_executor/model_loader/oci_go && ./build.sh")
        else:
            raise
except ImportError as e:
    if "torch" in str(e):
        print("⚠ Can't fully test (torch not installed), but wrapper code is valid")
    else:
        raise
EOF
```

### 5. Run Unit Tests

```bash
cd /path/to/vllm
python3 -m pytest tests/model_executor/model_loader/test_oci_loader.py -v
```

**Expected:** Tests pass or skip if dependencies missing

## Integration Test Scenarios

### Scenario 1: Public Image (No Authentication)

```python
from vllm import LLM

# This should work without authentication
llm = LLM(
    model="aistaging/smollm2-vllm:latest",
    load_format="oci"
)

# Generate to verify it works
output = llm.generate(["Hello"], max_tokens=10)
print(output)
```

### Scenario 2: Private Image (With Authentication)

```bash
# First authenticate
docker login ghcr.io
# Enter credentials
```

```python
from vllm import LLM

# This requires authentication
llm = LLM(
    model="ghcr.io/your-org/your-private-model:v1.0",
    load_format="oci"
)
```

### Scenario 3: Different Registry

```bash
# Authenticate with another registry
docker login registry.example.com
```

```python
from vllm import LLM

llm = LLM(
    model="registry.example.com/namespace/model:tag",
    load_format="oci"
)
```

## Troubleshooting

### Issue: "OCI Go library not found"

**Cause:** Go library not built

**Solution:**
```bash
cd vllm/model_executor/model_loader/oci_go
./build.sh
```

### Issue: "go: command not found"

**Cause:** Go not installed

**Solution:**
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install golang-1.24

# On macOS
brew install go

# Or download from https://go.dev/dl/
```

### Issue: "Failed to pull manifest: ... authentication required"

**Cause:** Image is private and not authenticated

**Solution:**
```bash
docker login [registry]
# Enter credentials
```

### Issue: "Failed to pull manifest: ... connection refused"

**Cause:** Network issue or wrong registry

**Solution:**
- Check network connectivity
- Verify registry URL is correct
- Check if registry requires VPN

### Issue: Build fails with "CGO_ENABLED not set"

**Cause:** CGO disabled

**Solution:**
```bash
export CGO_ENABLED=1
cd vllm/model_executor/model_loader/oci_go
./build.sh
```

### Issue: "invalid pointer" or segmentation fault

**Cause:** Memory management issue

**Solution:** This should be fixed in the current version. If you see this:
1. Rebuild the library: `./build.sh`
2. Report the issue with details

## Verification Checklist

Use this checklist to verify the integration:

- [ ] Go compiler is installed and accessible
- [ ] Go library builds successfully (liboci.so or liboci.dylib)
- [ ] Standalone tests pass
- [ ] Python wrapper imports successfully
- [ ] Can pull manifest from public registry
- [ ] Authentication works with docker login
- [ ] Can load a model from OCI registry (end-to-end test)
- [ ] Error messages are clear and helpful
- [ ] No memory leaks or crashes

## Performance Verification

### Build Performance

```bash
# Time the build
time ./build.sh

# Expected:
# - First build: 30-60 seconds
# - Subsequent builds: 5-10 seconds
```

### Runtime Performance

```bash
# Measure library load time
python3 << 'EOF'
import time
import sys
sys.path.insert(0, '/path/to/vllm')

start = time.time()
from vllm.model_executor.model_loader.oci_go_client import OciGoClient
client = OciGoClient()
load_time = time.time() - start

print(f"Library load time: {load_time*1000:.2f}ms")
# Expected: <100ms
EOF

# Measure manifest pull time
python3 << 'EOF'
import time
import sys
sys.path.insert(0, '/path/to/vllm')
from vllm.model_executor.model_loader.oci_go_client import OciGoClient

client = OciGoClient()

start = time.time()
manifest = client.pull_manifest("docker.io/library/alpine:latest")
pull_time = time.time() - start

print(f"Manifest pull time: {pull_time*1000:.2f}ms")
# Expected: 200-500ms (network dependent)
EOF
```

## Security Verification

### Check File Permissions

```bash
# Docker config should be readable only by user
ls -l ~/.docker/config.json
# Should show: -rw------- (600)

# If not, fix it:
chmod 600 ~/.docker/config.json
```

### Verify TLS/HTTPS

The library always uses HTTPS. To verify:

```bash
# This should work
python3 -c "
from vllm.model_executor.model_loader.oci_go_client import OciGoClient
client = OciGoClient()
manifest = client.pull_manifest('docker.io/library/alpine:latest')
print('✓ TLS works')
"
```

### Test Credential Helper

If using a credential helper:

```bash
# Check if helper is configured
cat ~/.docker/config.json | grep credHelpers

# Test the helper
docker-credential-ecr-login list  # For ECR
# or
docker-credential-gcr list  # For GCR
```

## Continuous Integration

For CI/CD pipelines:

```bash
# Install Go
curl -OL https://go.dev/dl/go1.24.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.24.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# Build vLLM (includes Go library)
pip install -e .

# Run tests
pytest tests/model_executor/model_loader/test_oci_loader.py
```

## Success Criteria

The integration is working correctly if:

1. ✓ Go library builds without errors
2. ✓ Standalone tests pass (3/3)
3. ✓ Can pull manifests from public registries
4. ✓ Docker login authentication works
5. ✓ No memory leaks or crashes
6. ✓ Error messages are clear
7. ✓ Performance is acceptable (<100ms load, <500ms network ops)
8. ✓ Works on target platforms (Linux, macOS)

## Getting Help

If verification fails:

1. Check this guide for troubleshooting steps
2. Review INTEGRATION_NOTES.md for detailed information
3. Check the logs for error messages
4. Run standalone tests to isolate the issue
5. Report issues with:
   - Error messages
   - Platform (OS, architecture)
   - Go version
   - Steps to reproduce
