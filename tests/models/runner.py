from PIL import Image
import torch
import numpy as np
# Load an image using Pillow
image = Image.open('img/flower.jpg')

# Convert the image to RGB (if it's not already in this mode)
image = image.convert('RGB')

# Convert the image to a tensor
tensor = torch.tensor(np.array(image))

# Normalize the pixel values (Optional, depends on your use case)
tensor = tensor.float() / 255.0  # Scale pixel values to [0, 1]i
img_tensor = tensor.permute(2, 0, 1)
print(img_tensor.shape)
img_tensor = img_tensor.unsqueeze(0)
print(img_tensor.shape)
torch.save(img_tensor, 'flower.pt')