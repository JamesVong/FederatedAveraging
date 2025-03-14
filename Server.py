# Server.py
import copy
import torch
from torch.utils.data import DataLoader

class Server:
    def __init__(self, model, val_dataset):
        self.model = model             # Global model (TwoLayerNet instance)
        self.val_dataset = val_dataset # Validation dataset for evaluation
        self.val_acc_history = []      # List to record global validation accuracy per round

    def aggregate(self, client_results):
        """
        Performs weighted averaging of client model parameters.
        client_results: dict mapping client name to (state_dict, sample_count, train_acc)
        """
        total_samples = sum([client_results[key][1] for key in client_results])
        # Start with a copy of the first client's state_dict
        new_state = copy.deepcopy(client_results[list(client_results.keys())[0]][0])
        # Weighted contribution from the first client
        weight = client_results[list(client_results.keys())[0]][1] / total_samples
        for key in new_state.keys():
            new_state[key] = new_state[key] * weight
        # Sum contributions from remaining clients
        for client in list(client_results.keys())[1:]:
            client_state, count, _ = client_results[client]
            weight = count / total_samples
            for key in new_state.keys():
                new_state[key] += client_state[key] * weight
        self.model.load_state_dict(new_state)

    def evaluate(self, device, batch_size=64):
        """
        Evaluate the current global model on the validation set.
        """
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = correct / total
        return acc
