# Custom RGBA Background Color Feature

This feature allows users to specify custom background colors when converting RGBA images to RGB format in vLLM.

## Usage

### Via Python API

```python
from vllm.multimodal.image import ImageMediaIO

# Default white background (backward compatible)
image_io = ImageMediaIO()

# Custom black background
image_io = ImageMediaIO(rgba_background_color=(0, 0, 0))

# Custom blue background (can use list or tuple)
image_io = ImageMediaIO(rgba_background_color=[0, 0, 255])

# Load and convert an RGBA image
image = image_io.load_file("path/to/rgba_image.png")
```

### Via CLI

You can specify custom RGBA background colors using the `--media-io-kwargs` parameter:

```bash
# Black background
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'

# Custom color (R=128, G=128, B=128)
python -m vllm.entrypoints.openai.api_server \
    --model your-model \
    --media-io-kwargs '{"image": {"rgba_background_color": [128, 128, 128]}}'
```

## Use Cases

1. **Dark Theme Support**: Use black background for better integration with dark UIs
2. **Brand Consistency**: Match your application's color scheme
3. **Model Requirements**: Some vision models may perform better with specific background colors
4. **Transparency Handling**: Control how transparent pixels are rendered

## Technical Details

- **Default Behavior**: If not specified, white (255, 255, 255) is used as the background
- **Backward Compatibility**: Existing code continues to work without changes
- **Format Support**: Accepts both tuple `(R, G, B)` and list `[R, G, B]` formats
- **Value Range**: RGB values should be integers from 0 to 255

## Example

```python
from PIL import Image
from vllm.multimodal.image import ImageMediaIO

# Create an RGBA image with transparency
rgba_image = Image.new("RGBA", (100, 100), (255, 0, 0, 128))  # Semi-transparent red

# Save it
rgba_image.save("test.png")

# Load with different backgrounds
io_white = ImageMediaIO()  # Default white
io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))

img_white = io_white.load_file("test.png")  # Transparent areas become white
img_black = io_black.load_file("test.png")  # Transparent areas become black
img_green = io_green.load_file("test.png")  # Transparent areas become green
```