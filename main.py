import random
import torch
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
from utils import get_dataset, split_dataset_dirichlet
from client import client_update
from server import server_aggregate

def main():
    dataset_name = 'mnist'
    num_clients = int(input("Enter number of clients: ").strip())
    alpha = float(input("Enter Dirichlet alpha (e.g. 0.5): ").strip())
    num_rounds = int(input("Enter number of communication rounds: ").strip())
    local_epochs = int(input("Enter number of local epochs per client: ").strip())
    num_attackers = int(input("Enter the number of attackers: ").strip())
    
    # Randomly select attacker client IDs from the available client pool
    if num_attackers > 0:
        malicious_clients = random.sample(range(num_clients), num_attackers)
    else:
        malicious_clients = []
    print(f"Malicious client IDs: {malicious_clients}")
    
    # Load the dataset and split it among clients using Dirichlet distribution
    trainset, testset = get_dataset(dataset_name)
    client_indices = split_dataset_dirichlet(trainset, num_clients, alpha)
    
    # Set number of channels based on dataset (MNIST: 1, CIFAR-10: 3)
    in_channels = 1 if dataset_name == 'mnist' else 3
    global_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    
    # Begin federated learning rounds
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")
        client_state_dicts = []
        
        for client_id in range(num_clients):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            # Initialise the local model for the client with the current global weights
            local_model = SimpleCNN(num_classes=10, in_channels=in_channels)
            local_model.load_state_dict(global_model.state_dict())
            
            # Check if this client is designated as malicious based on the random selection
            is_malicious = client_id in malicious_clients
            local_state = client_update(local_model, trainloader, local_epochs=local_epochs, malicious=is_malicious)
            client_state_dicts.append(local_state)
        
        # Aggregate the local updates into the global model
        global_model = server_aggregate(global_model, client_state_dicts)
    
    # Evaluate the global model on the test set
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            outputs = global_model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f"\nGlobal model accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
