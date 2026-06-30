
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.data import get_cifar10_dataloaders
from src.model import VisionTransformer
from config import LR, NUM_EPOCHS

def train():
    train_loader, test_loader = get_cifar10_dataloaders()
    model = VisionTransformer()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            predictions =model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()