import argparse
import torch
from torch.utils.data import DataLoader, Subset

from models import SimpleCNN
from utils import get_dataset, split_dataset_dirichlet
from client import client_update
from server import server_aggregate

def main(dataset_name='mnist', num_clients=10, alpha=0.5, num_rounds=5, local_epochs=1):
    # Load the dataset
    trainset, testset = get_dataset(dataset_name)
    client_indices = split_dataset_dirichlet(trainset, num_clients, alpha)
    
    # Set input channels based on dataset (MNIST: 1, CIFAR-10: 3)
    in_channels = 1 if dataset_name.lower() == 'mnist' else 3
    global_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")
        client_state_dicts = []
        
        for client_id in range(num_clients):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            # Create a local model for the client and initialise it with the global weights
            local_model = SimpleCNN(num_classes=10, in_channels=in_channels)
            local_model.load_state_dict(global_model.state_dict())
            local_state = client_update(local_model, trainloader, local_epochs=local_epochs)
            client_state_dicts.append(local_state)
        
        # Aggregate local updates into the global model
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
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument('--dataset', type=str, default='mnist', help="Dataset: 'mnist' or 'cifar10'")
    parser.add_argument('--num_clients', type=int, default=10, help="Number of clients")
    parser.add_argument('--alpha', type=float, default=0.5, help="Dirichlet alpha for non-IID split")
    parser.add_argument('--num_rounds', type=int, default=5, help="Number of communication rounds")
    parser.add_argument('--local_epochs', type=int, default=1, help="Local epochs per client")
    args = parser.parse_args()
    
    main(dataset_name=args.dataset, num_clients=args.num_clients, alpha=args.alpha, 
         num_rounds=args.num_rounds, local_epochs=args.local_epochs)
