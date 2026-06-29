from src.data import get_cifar10_dataloaders
from src.patches import PatchEmbedding
import torch

def main():
    train_loader, test_loader = get_cifar10_dataloaders()
    print("Data loaders created successfully.")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Quick sanity check for the PatchEmbedding module
    patch_embedding = PatchEmbedding()
    print("Patch embedding module created successfully.")
    x = torch.randn(8, 3, 32, 32) # Example input tensor of same shape as CIFAR-1O images and batch size of 8
    output = patch_embedding(x)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
