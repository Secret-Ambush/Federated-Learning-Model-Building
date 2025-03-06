import random
import torch
from torch.utils.data import DataLoader, Subset
from models import SimpleCNN
from utils import get_dataset, split_dataset_dirichlet
from client import client_update
from server import server_aggregate
from backdoor import inject_backdoor_dynamic, plot_backdoor_images

def main():
    # --- Interactive Prompts ---
    dataset_name = input("Choose dataset ('mnist' or 'cifar10'): ").strip().lower()
    
    num_clients = int(input("Enter number of clients: ").strip())
    alpha = float(input("Enter Dirichlet alpha (e.g. 0.5): ").strip())
    num_rounds = int(input("Enter number of communication rounds: ").strip())
    local_epochs = int(input("Enter number of local epochs per client: ").strip())
    
    num_attackers = int(input("Enter the number of attackers: ").strip())
    if num_attackers > 0:
        malicious_clients = random.sample(range(num_clients), num_attackers)
    else:
        malicious_clients = []
    print(f"Selected malicious client IDs: {malicious_clients}")
    
    # Backdoor parameters:
    inj_rate = float(input("Enter injection rate for backdoor on malicious clients (e.g. 0.5 for 50%): ").strip())
    pat_size = float(input("Enter pattern size as fraction of image (e.g. 0.1 for 10%): ").strip())
    loc = input("Enter pattern placement ('fixed' or 'random'): ").strip().lower()
    target_label = int(input("Enter target label for the attack (e.g. 1): ").strip())
    # We fix the pattern type as 'plus'
    pattern_type = "plus"
    
    # --- Data Preparation ---
    trainset, testset = get_dataset(dataset_name)
    client_indices = split_dataset_dirichlet(trainset, num_clients, alpha)
    
    # Set in_channels based on dataset (for CIFAR10, 3 channels)
    in_channels = 1 if dataset_name == 'mnist' else 3
    global_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    
    # --- Federated Learning Training ---
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd + 1} ---")
        client_state_dicts = []
        
        for client_id in range(num_clients):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            # Initialize local model with global weights
            local_model = SimpleCNN(num_classes=10, in_channels=in_channels)
            local_model.load_state_dict(global_model.state_dict())
            
            # Determine if this client is malicious
            is_malicious = client_id in malicious_clients
            
            # For demonstration: output sample backdoor images for the first malicious client in round 1.
            if rnd == 0 and is_malicious:
                sample_data, sample_target = next(iter(trainloader))
                # Apply backdoor to 100% of the sample for visualization.
                sample_data_bd, _ = inject_backdoor_dynamic(sample_data.clone(), sample_target.clone(),
                                                            injection_rate=1.0,
                                                            pattern_type=pattern_type,
                                                            pattern_size=pat_size,
                                                            location=loc,
                                                            target_label=target_label)
                print("Displaying sample backdoor images from a malicious client:")
                plot_backdoor_images(sample_data_bd, n=8)
            
            # Update local model (with backdoor injection if malicious)
            local_state = client_update(local_model, trainloader, local_epochs=local_epochs, lr=0.01,
                                        malicious=is_malicious,
                                        injection_rate=inj_rate,
                                        pattern_size=pat_size,
                                        location=loc,
                                        pattern_type=pattern_type,
                                        target_label=target_label)
            client_state_dicts.append(local_state)
        
        # Aggregate local models into global model
        global_model = server_aggregate(global_model, client_state_dicts)
    
    # --- Evaluation on Clean Test Set ---
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
    print(f"\nGlobal model accuracy on clean test set: {100 * correct / total:.2f}%")
    
    # --- Evaluation on Backdoor Test Set ---
    # Inject backdoor into 100% of the test images.
    testloader_bd = DataLoader(testset, batch_size=32, shuffle=False)
    total_bd = 0
    target_pred_bd = 0
    with torch.no_grad():
        for data, target in testloader_bd:
            data_bd, _ = inject_backdoor_dynamic(data.clone(), target.clone(),
                                                 injection_rate=1.0,
                                                 pattern_type=pattern_type,
                                                 pattern_size=pat_size,
                                                 location=loc,
                                                 target_label=target_label)
            outputs = global_model(data_bd)
            _, predicted = torch.max(outputs, 1)
            total_bd += data_bd.size(0)
            target_pred_bd += (predicted == target_label).sum().item()
    attack_success_rate = 100 * target_pred_bd / total_bd
    print(f"Attack success rate on backdoor test set: {attack_success_rate:.2f}%")

if __name__ == "__main__":
    main()
