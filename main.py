from src.data import get_cifar10_dataloaders

def main():
    train_loader, test_loader = get_cifar10_dataloaders()
    print("Data loaders created successfully.")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()
