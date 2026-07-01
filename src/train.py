
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.data import get_cifar10_dataloaders
from src.model import VisionTransformer
from config import LR, NUM_EPOCHS

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_cifar10_dataloaders()
    model = VisionTransformer().to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()