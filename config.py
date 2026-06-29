
# Configuration file containing all hyperparameters and setting for model training and evaluation

# Data parameters
BATCH_SIZE = 64
IMG_SIZE = 32                                    # Image dimensions (technically not a hyperparameter)
NUM_CLASSES = 10                                 # (technically not a hyperparameter)

# Patch embedding parameters
PATCH_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2     # Number of patches depending on image and patch size
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3         # flattened patch size


# Transformer model parmeters
D_MODEL = 128                                   # Embedding dimension
NUM_HEADS = 2                                   # Number of attention heads
NUM_LAYERS = 6
MLP_DIM = 256
DROPOUT = 0.1


# Training parameters
LR = 1e-3 
NUM_EPOCHS = 10

