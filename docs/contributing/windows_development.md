# Windows Development Guide

This guide covers Windows-specific considerations for vLLM development, including common issues and solutions.

## Documentation Building on Windows

### Encoding Issues

Windows uses different default encodings than Unix systems, which can cause issues when building documentation locally.

#### Common Symptoms

- `UnicodeEncodeError: 'charmap' codec can't encode character`
- Documentation build failures with encoding-related errors
- Issues with Unicode characters in documentation files

#### Solutions

**Method 1: Set UTF-8 Environment Variables**

```powershell
# In PowerShell, set UTF-8 encoding for the session
$env:PYTHONUTF8=1
chcp 65001

# Then run your documentation build
mkdocs serve
```

**Method 2: Set UTF-8 Permanently**

1. Open System Properties → Advanced → Environment Variables
2. Add a new system variable:
   - Name: `PYTHONUTF8`
   - Value: `1`
3. Restart your terminal

**Method 3: Use UTF-8 Code Page**

```cmd
# In Command Prompt
chcp 65001
mkdocs serve
```

### Building Documentation Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements/docs.txt
   ```

2. **Set UTF-8 Encoding** (choose one method above)

3. **Build Documentation**
   ```bash
   # Serve locally with hot reload
   mkdocs serve
   
   # Build static files
   mkdocs build
   ```

## Common Windows Development Issues

### Path Separators

Windows uses backslashes (`\`) while Unix systems use forward slashes (`/`). Use `pathlib.Path` for cross-platform compatibility:

```python
# Good - cross-platform
from pathlib import Path
file_path = Path("docs") / "examples" / "file.md"

# Avoid - platform-specific
file_path = "docs\\examples\\file.md"  # Windows only
file_path = "docs/examples/file.md"    # Unix only
```

### File Encoding

Always specify encoding when opening files to avoid platform-specific defaults:

```python
# Good - explicit encoding
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Avoid - platform-dependent default encoding
with open(file_path, 'r') as f:  # May use cp1252 on Windows
    content = f.read()
```

## Best Practices for Cross-Platform Development

### File Operations

```python
# Always specify encoding
with open(path, 'r', encoding='utf-8') as f:
    data = f.read()

# Use pathlib for paths
from pathlib import Path
config_path = Path.home() / '.vllm' / 'config.json'

# Handle temporary files properly
import tempfile
with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
    f.write(data)
    temp_path = Path(f.name)
```

### Environment Variables

```python
import os
from pathlib import Path

# Cross-platform home directory
home_dir = Path.home()

# Environment variables with defaults
cache_dir = os.getenv('VLLM_CACHE_DIR', str(home_dir / '.cache' / 'vllm'))
```

## Troubleshooting

### Documentation Build Fails

1. **Check Python Version**: Ensure Python 3.9+ is installed
2. **Set UTF-8 Encoding**: Use methods described above
3. **Check Dependencies**: Ensure all requirements are installed
4. **Clear Cache**: Delete `site/` directory and rebuild

### Import Errors

1. **Check PYTHONPATH**: Ensure vLLM is in Python path
2. **Virtual Environment**: Use a clean virtual environment
3. **Dependencies**: Install all required dependencies

## Contributing from Windows

1. **Line Endings**: Configure Git to handle line endings:
   ```bash
   git config --global core.autocrlf true
   ```

2. **File Permissions**: Windows doesn't have Unix file permissions, so permission-related tests may behave differently

3. **Testing**: Always test changes on both Windows and Unix systems when possible

4. **Documentation**: Include Windows-specific instructions when adding new features

Remember: When in doubt, always specify encoding explicitly and use cross-platform libraries like `pathlib` for file operations.
