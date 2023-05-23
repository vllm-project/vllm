# CacheFlow documents

## Build the docs

```bash
# Install dependencies.
pip -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your brower

```bash
cd build/html
python -m http.server
```
Launch your browser and open localhost:8000.
