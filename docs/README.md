# vLLM documents

## Build the docs

- Make sure in `docs` directory

```bash
cd docs
```

- Install the dependencies:

```bash
pip install -r ../requirements/docs.txt
```

- Clean the previous build (optional but recommended):

```bash
make clean
```

- Generate the HTML documentation:

```bash
make html
```

## Open the docs with your browser

- Serve the documentation locally:

```bash
python -m http.server -d build/html/
```

This will start a local server at http://localhost:8000. You can now open your browser and view the documentation.

If port 8000 is already in use, you can specify a different port, for example:

```bash
python -m http.server 3000 -d build/html/
```
