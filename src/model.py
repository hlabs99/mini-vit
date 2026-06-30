import torch
import torch.nn as nn


from src.patches import PatchEmbedding
from src.encoder import EncoderBlock
from config import D_MODEL, NUM_LAYERS, NUM_CLASSES

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = PatchEmbedding()
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        x = self.patch_embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        # x has shape (batch_size, num_patches, D_MODEL), we take the first token (class token) for classification
        x = self.classifier(x[:, 0, :])
        # x now has shape (batch_size, NUM_CLASSES). For each class, we have a score (logit).
        # We can apply softmax to get probabilities, but for training with CrossEntropyLoss, we can 
        # return the logits directly because CrossEntropyLoss expects raw logits.
        return x