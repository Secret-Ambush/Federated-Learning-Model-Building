import argparse
import torch
from torch.utils.data import DataLoader, Subset

from models import SimpleCNN
from utils import get_dataset, split_dataset_dirichlet
from client import client_update
from server import server_aggregate

def main(dataset_name='mnist', num_clients=10, alpha=0.5, num_rounds=5, local_epochs=1, malicious_clients=None):
    # Load the dataset
    trainset, testset = get_dataset(dataset_name)
    client_indices = split_dataset_dirichlet(trainset, num_clients, alpha)
    
    # For MNIST: 1 channel, for CIFAR-10: 3 channels
    in_channels = 1 if dataset_name.lower() == 'mnist' else 3
    global_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    
    # Define malicious clients (if not provided, default to an empty list)
    if malicious_clients is None:
        malicious_clients = []  # e.g. [0, 1] for the first two clients as attackers
    
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")
        client_state_dicts = []
        
        for client_id in range(num_clients):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            # Initialise the local model with the global weights
            local_model = SimpleCNN(num_classes=10, in_channels=in_channels)
            local_model.load_state_dict(global_model.state_dict())
            
            # Determine if the current client is malicious
            is_malicious = client_id in malicious_clients
            local_state = client_update(local_model, trainloader, local_epochs=local_epochs, malicious=is_malicious)
            client_state_dicts.append(local_state)
        
        # Aggregate updates from all clients
        global_model = server_aggregate(global_model, client_state_dicts)
    
    # Evaluate the global model
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
    parser = argparse.ArgumentParser(description="Federated Learning Simulation with Backdoor Attack")
    parser.add_argument('--dataset', type=str, default='mnist', help="Dataset: 'mnist' or 'cifar10'")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--alpha', type=float, default=0.5, help="Dirichlet alpha for non-IID split")
    parser.add_argument('--num_rounds', type=int, default=5, help="Number of communication rounds")
    parser.add_argument('--local_epochs', type=int, default=1, help="Local epochs per client")
    parser.add_argument('--malicious_clients', type=str, default="", 
                        help="Comma-separated list of malicious client IDs (e.g. '0,1')")
    args = parser.parse_args()
    
    # Parse the malicious clients argument into a list of integers
    malicious_clients = list(map(int, args.malicious_clients.split(','))) if args.malicious_clients else []
    
    main(dataset_name=args.dataset, num_clients=args.num_clients, alpha=args.alpha, 
         num_rounds=args.num_rounds, local_epochs=args.local_epochs, malicious_clients=malicious_clients)
