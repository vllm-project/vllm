# LMCache Basic Check Examples

This is an introduce of examples for the LMCache Basic Check Tool.

## Example Usage Patterns

### Testing Storage Manager
```bash
# Basic test
python -m lmcache.v1.basic_check --mode test_storage_manager

# With custom model
python -m lmcache.v1.basic_check --mode test_storage_manager --model /my_model/
```

### Testing Remote Backend
```bash
# Basic remote test
python -m lmcache.v1.basic_check --mode test_remote

```

### Key Generation
```bash
# Generate 100 keys with 8 concurrent workers
python -m lmcache.v1.basic_check --mode gen --num-keys 100 --concurrency 8

# Generate keys with offset (useful for distributed testing)
python -m lmcache.v1.basic_check --mode gen --num-keys 100 --concurrency 8 --offset 1000
```

## Configuration

Use the provided example configuration:

```bash
# Copy to default location
cp example_config.yaml ~/.lmcache/config.yaml

# Or set environment variable
export LMCACHE_CONFIG_PATH=$(pwd)/example_config.yaml
```

## Documentation

For comprehensive documentation, see:
- [Detailed Usage Documentation](../../docs/source/developer_guide/usage/basic_check.rst)
