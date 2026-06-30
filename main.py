import torch

from src.data import get_cifar10_dataloaders
from src.patches import PatchEmbedding
from src.attention import MultiHeadAttention
from src.encoder import EncoderBlock
from src.model import VisionTransformer
from src.train import train


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

    # Test attention
    attention = MultiHeadAttention()
    output = attention(output)
    print(f"MultiHeadAttention output shape: {output.shape}")
    
    # Check attention weights are reasonable
    print(f"Attention output mean: {output.mean().item():.4f}")
    print(f"Attention output std: {output.std().item():.4f}")

    # Test the Encoder Block
    encoder_block = EncoderBlock()
    output = encoder_block(output)
    print(f"EncoderBlock output shape: {output.shape}")

    # Test the full Vision Transformer model
    model = VisionTransformer()
    output = model(x)
    print(f"VisionTransformer output shape: {output.shape}")

    # Sanity check for training loop
    print("Starting training loop...")
    train()  # This will run the training loop defined in train.py
    


    
if __name__ == "__main__":
    main()
