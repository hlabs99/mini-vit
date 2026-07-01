import torch

from src.data import get_cifar10_dataloaders
from src.model import VisionTransformer


def evaluate():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")


    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))

    model.eval()

    train_loader, test_loader = get_cifar10_dataloaders()

    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predicted = predictions.argmax(dim=1) 
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    evaluate()

