import torch

# Create a randomly initialized 5x3 tensor
x = torch.rand(5, 3)
print("Random Tensor:\n", x)

# Check if CUDA is available and print the result
cuda_available = torch.cuda.is_available()
print("\nCUDA available:", cuda_available)

# If CUDA is available, you can also try moving a tensor to the GPU
if cuda_available:
    device = torch.device("cuda")
    y = torch.ones(2, 2, device=device)
    print("\nTensor on GPU:\n", y)