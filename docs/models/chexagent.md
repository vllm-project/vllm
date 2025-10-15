# CheXagent Model

CheXagent is a multimodal model specifically designed for medical image analysis and interpretation. It integrates a QFormer architecture to process medical images and generate detailed medical reports.

## Model Architecture

CheXagent consists of three main components:

1. **Vision Model**: Processes medical images using a BLIP vision encoder
2. **QFormer**: A query-based transformer that bridges vision and language modalities
3. **Language Model**: Generates medical text based on the processed image features

## Usage

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="StanfordAIMI/CheXagent-8b",
    trust_remote_code=True,
    dtype="auto"
)

# Prepare your prompt with an image
prompt = "<image> Describe the findings in this chest X-ray."

# Generate response
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate([prompt], sampling_params, multi_modal_data={"image": [image_path]})

print(outputs[0].outputs[0].text)
```

### Using with vLLM API

```python
import requests
import base64

# Encode your image
with open("chest_xray.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the request
data = {
    "model": "StanfordAIMI/CheXagent-8b",
    "prompt": "<image> Analyze this chest X-ray and provide a detailed report.",
    "max_tokens": 512,
    "temperature": 0.7,
    "multi_modal_data": {
        "image": [encoded_image]
    }
}

# Send request to vLLM API
response = requests.post("http://localhost:8000/v1/completions", json=data)
print(response.json()["choices"][0]["text"])
```

## Model Configuration

The CheXagent model supports the following configuration options:

- `num_query_tokens`: Number of query tokens for the QFormer (default: 32)
- `vision_config`: Configuration for the vision encoder
- `qformer_config`: Configuration for the QFormer transformer
- `text_config`: Configuration for the language model

## Supported Image Formats

CheXagent supports standard image formats:
- JPEG
- PNG
- BMP
- TIFF

## Medical Use Cases

CheXagent is particularly well-suited for:

1. **Chest X-ray Analysis**: Detecting pneumonia, tuberculosis, and other lung conditions
2. **Medical Report Generation**: Creating detailed radiology reports
3. **Medical Image Interpretation**: Explaining findings in medical images
4. **Medical Education**: Teaching medical students about image interpretation

## Performance Considerations

- **Memory Usage**: The model requires significant GPU memory due to the multimodal architecture
- **Batch Processing**: Supports batch processing of multiple images
- **Quantization**: Compatible with vLLM's quantization features for reduced memory usage

## Limitations

1. **Medical Disclaimer**: This model is for research and educational purposes only
2. **Not for Clinical Use**: Should not be used for actual clinical decision-making
3. **Image Quality**: Performance may vary with image quality and resolution
4. **Domain Specificity**: Optimized for medical images, particularly chest X-rays

## Citation

If you use CheXagent in your research, please cite:

```bibtex
@article{chexagent2024,
  title={CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation},
  author={...},
  journal={...},
  year={2024}
}
```

## Troubleshooting

### Common Issues

1. **"Model architecture not supported"**: Ensure you're using the latest version of vLLM
2. **Memory errors**: Try reducing batch size or using quantization
3. **Image loading issues**: Check image format and file path

### Getting Help

- Check the [vLLM documentation](https://docs.vllm.ai/)
- Report issues on the [vLLM GitHub repository](https://github.com/vllm-project/vllm)
- For CheXagent-specific issues, refer to the original model repository 