import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

###############################
# 1. Define the Two-Layer Perceptron Model
###############################

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, output_dim=10):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # For images, flatten the input tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

###############################
# 2. Utility functions for training and evaluation
###############################

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        running_correct += (predicted == targets).sum().item()
    accuracy = running_correct / total
    return accuracy

def evaluate(model, dataloader, device):
    model.eval()
    running_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            running_correct += (predicted == targets).sum().item()
    accuracy = running_correct / total
    return accuracy

###############################
# 3. Main function for baseline training
###############################

def main():
    # Choose dataset: "FashionMNIST" or "CIFAR10"
    dataset_name = "FashionMNIST"
    
    # Hyperparameters
    num_repeats = 5
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    hidden_dim = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up dataset and determine input dimension based on dataset
    if dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        input_dim = 28 * 28
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_dim = 32 * 32 * 3
    else:
        raise ValueError("Dataset not supported. Choose 'FashionMNIST' or 'CIFAR10'.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    final_test_accuracies = []

    # Repeat the training process multiple times
    for repeat in range(num_repeats):
        print(f"\n=== Repeat {repeat+1}/{num_repeats} ===")
        # Initialize model, criterion and optimizer
        model = TwoLayerNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        train_acc_history = []
        test_acc_history = []
        
        for epoch in range(num_epochs):
            train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        final_test_accuracies.append(test_acc_history[-1])
        
        # Plot training and test accuracy curves for this repeat
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, num_epochs+1), train_acc_history, marker='o', label="Train Accuracy")
        plt.plot(range(1, num_epochs+1), test_acc_history, marker='o', label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Baseline Accuracy Curves (Repeat {repeat+1}) on {dataset_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Report final accuracy statistics
    mean_final_acc = np.mean(final_test_accuracies)
    std_final_acc = np.std(final_test_accuracies)
    print(f"\nFinal Test Accuracy over {num_repeats} repeats:")
    print(f"Mean: {mean_final_acc:.4f}, Standard Deviation: {std_final_acc:.4f}")

if __name__ == '__main__':
    main()
