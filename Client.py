# Client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Client:
    def __init__(self, name, model=None, dataset=None):
        self.name = name
        self.model = model              # A TwoLayerNet instance (architecture only)
        self.dataset = dataset          # Local subset (non-overlapping)
        self.train_acc_history = []     # List to record local training accuracy per round

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset):
        self.dataset = dataset

    def train_local_model(self, global_state, local_epochs, lr, batch_size, device, return_dict):
        """
        Loads the global model weights, trains on the local dataset for a given
        number of local epochs, and then returns the updated weights along with
        the number of samples and the final local training accuracy for the round.
        """
        # Create a new instance of the model (using the same architecture)
        local_model = type(self.model)(
            input_dim=self.model.fc1.in_features,
            hidden_dim=self.model.fc1.out_features,
            output_dim=self.model.fc2.out_features
        ).to(device)
        local_model.load_state_dict(global_state)
        
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        
        # Train for local_epochs (here we use one epoch per communication round)
        for epoch in range(local_epochs):
            local_model.train()
            correct = 0
            total = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            round_acc = correct / total
        # Record this round's local training accuracy
        self.train_acc_history.append(round_acc)
        # Place in return_dict: (state_dict, number of samples, training accuracy)
        return_dict[self.name] = (local_model.state_dict(), len(self.dataset), round_acc)
