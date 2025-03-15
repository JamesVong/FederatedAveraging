# main.py
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from multiprocessing import Process, Manager, set_start_method
from Client import Client
from Server import Server
from Baseline import TwoLayerNet  # Assumes TwoLayerNet is defined in Baseline.py

# Ensure we use spawn to avoid CUDA issues
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def partition_dataset(dataset, num_clients=5, uneven=True):
    """
    Partition the dataset indices unevenly by class.
    Returns a dict mapping client index to a list of indices.
    """
    num_classes = len(dataset.classes)
    idx_by_class = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        idx_by_class[label].append(idx)
    client_indices = {i: [] for i in range(num_clients)}
    for cls in idx_by_class:
        indices = idx_by_class[cls]
        np.random.shuffle(indices)
        # Use a Dirichlet distribution for uneven splits
        proportions = np.random.dirichlet(np.ones(num_clients), size=1)[0]
        splits = (proportions * len(indices)).astype(int)
        diff = len(indices) - np.sum(splits)
        for i in range(diff):
            splits[i % num_clients] += 1
        start = 0
        for i in range(num_clients):
            end = start + splits[i]
            client_indices[i].extend(indices[start:end])
            start = end
    return client_indices

def main():
    # Settings
    dataset_name = "CIFAR10"
    num_clients = 5
    num_rounds = 10       # Communication rounds per repeat
    local_epochs = 1      # Local training epochs per round
    num_repeats = 5
    batch_size = 64
    learning_rate = 0.01
    hidden_dim = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # Prepare dataset: FashionMNIST with a train/validation split
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[Main] Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    
    # Partition full_dataset by label, then filter indices to those in train_dataset
    full_client_indices = partition_dataset(full_dataset, num_clients=num_clients, uneven=True)
    train_indices_set = set(train_dataset.indices)
    client_indices_filtered = {}
    for client_id, indices in full_client_indices.items():
        filtered = [idx for idx in indices if idx in train_indices_set]
        client_indices_filtered[client_id] = filtered
        print(f"[Main] Client {client_id} receives {len(filtered)} samples.")
    
    # Create client objects (each with its own non-overlapping subset)
    clients = []
    for client_id in range(num_clients):
        client_subset = Subset(full_dataset, client_indices_filtered[client_id])
        client = Client(name=f"Client_{client_id}")
        input_dim = 28 * 28  # For FashionMNIST
        model = TwoLayerNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=10)
        client.set_model(model)
        client.set_dataset(client_subset)
        clients.append(client)
    
    # Create server with a global model and the validation set
    global_model = TwoLayerNet(input_dim=28*28, hidden_dim=hidden_dim, output_dim=10).to(device)
    server = Server(model=global_model, val_dataset=val_dataset)
    
    # Create folder for saving FedAvg plots
    os.makedirs("FedAvgs", exist_ok=True)
    
    all_final_val_acc = []  # Record final validation accuracy for each repeat

    # Repeat the FedAvg training process
    for repeat in range(num_repeats):
        print(f"\n=== FedAvg Training Repeat {repeat+1}/{num_repeats} ===")
        # Reinitialize the global model and reset histories for each repeat
        global_model = TwoLayerNet(input_dim=28*28, hidden_dim=hidden_dim, output_dim=10).to(device)
        server.model = global_model
        for client in clients:
            client.train_acc_history = []  # Reset training history in the parent object
        server.val_acc_history = []
        
        # Communication rounds
        for rnd in range(num_rounds):
            print(f"\n[Main] Starting Communication Round {rnd+1}/{num_rounds}")
            global_state = server.model.state_dict()
            manager = Manager()
            return_dict = manager.dict()
            processes = []
            # Launch a process for each client; force training on CPU to avoid CUDA sharing issues
            for client in clients:
                print(f"[Main] Launching process for {client.name}")
                p = Process(target=client.train_local_model, args=(global_state, local_epochs, learning_rate, batch_size, torch.device('cpu'), return_dict))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            print(f"[Main] All client processes finished for round {rnd+1}. Return dict keys: {list(return_dict.keys())}")
            
            # Update parent's client training accuracy history from return_dict
            for client in clients:
                if client.name in return_dict:
                    round_acc = return_dict[client.name][2]
                    client.train_acc_history.append(round_acc)
                    print(f"[Main] Updated {client.name} training accuracy history: {client.train_acc_history}")
            
            # Aggregate client updates on the server
            server.aggregate(return_dict)
            # Evaluate global model on validation set
            val_acc = server.evaluate(device, batch_size)
            server.val_acc_history.append(val_acc)
            print(f"[Main] Server validation accuracy for round {rnd+1}: {val_acc:.4f}")
        
        all_final_val_acc.append(server.val_acc_history[-1])
        
        # Plot and save each client's training accuracy curve over rounds
        plt.figure(figsize=(8, 6))
        for client in clients:
            print(f"[Main] {client.name} final training accuracy history: {client.train_acc_history}")
            plt.plot(range(1, num_rounds+1), client.train_acc_history, marker='o', label=client.name)
        plt.xlabel("Communication Round")
        plt.ylabel("Local Training Accuracy")
        plt.title(f"{dataset_name} Client Training Accuracy (Repeat {repeat+1})")
        plt.legend()
        plt.grid(True)
        client_plot_filename = os.path.join("FedAvgs", f"{dataset_name}_ClientTraining_Repeat{repeat+1}.png")
        plt.savefig(client_plot_filename)
        plt.close()
        print(f"[Main] Saved client training plot: {client_plot_filename}")
        
        # Plot and save the server's validation accuracy curve over rounds
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, num_rounds+1), server.val_acc_history, marker='o', label="Server Validation Accuracy")
        plt.xlabel("Communication Round")
        plt.ylabel("Validation Accuracy")
        plt.title(f"{dataset_name} Server Validation Accuracy (Repeat {repeat+1})")
        plt.legend()
        plt.grid(True)
        server_plot_filename = os.path.join("FedAvgs", f"{dataset_name}_ServerValidation_Repeat{repeat+1}.png")
        plt.savefig(server_plot_filename)
        plt.close()
        print(f"[Main] Saved server validation plot: {server_plot_filename}")
    
    # Report final validation accuracy statistics over repeats
    mean_val_acc = np.mean(all_final_val_acc)
    std_val_acc = np.std(all_final_val_acc)
    print(f"\n[Main] Final Validation Accuracy over {num_repeats} repeats:")
    print(f"[Main] Mean: {mean_val_acc:.4f}, Standard Deviation: {std_val_acc:.4f}")

if __name__ == '__main__':
    main()
