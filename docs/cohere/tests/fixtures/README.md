# Test Fixtures

This directory contains test fixtures (images, data files, etc.) used by the Cohere test suite.

## Vision Test Images

The vision tests (`test_vision.py`) require two images that should be committed to this directory:

- **duck.jpg** - Image of a duck
- **lion.jpg** - Image of a lion

### Download Images

If the images are not present, run these commands from the repository root:

```bash
# Download duck image
curl -L "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg" \
  -o tests/cohere/fixtures/duck.jpg

# Download lion image
curl -L "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg" \
  -o tests/cohere/fixtures/lion.jpg
```

Or using wget:

```bash
# Download duck image
wget "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg" \
  -O tests/cohere/fixtures/duck.jpg

# Download lion image
wget "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg" \
  -O tests/cohere/fixtures/lion.jpg
```

### Image Sources

Both images are from Wikimedia Commons and are in the public domain:

- Duck: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2015_Kaczka_krzy%C5%BCowka_w_wodzie_(samiec).jpg)
- Lion: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg)

These images should be committed to the repository to:

- Avoid network dependencies during test runs
- Speed up test execution
- Ensure tests work in offline environments
