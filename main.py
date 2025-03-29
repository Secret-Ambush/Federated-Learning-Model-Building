import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from models import SimpleCNN
from utils import get_dataset, split_dataset_dirichlet
from client import client_update
from server import server_aggregate
from backdoor import inject_backdoor_dynamic

DATASET = "cifar10"        
NUM_CLIENTS = 20            
ALPHA = 0.5                
NUM_ROUNDS = 30             
LOCAL_EPOCHS = 5            

INJECTION_RATE = 0.5       
PATTERN_SIZE = 0.1         
TARGET_LABEL = 1         

# Attacker percentages to test (percentage of total clients).
ATTACKER_PERCENTAGES = [0, 10, 20, 30, 40, 50]

# Define different backdoor configurations.
configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"}
]

# ----- Prepare Dataset -----
trainset, testset = get_dataset()
client_indices = split_dataset_dirichlet(trainset, NUM_CLIENTS, ALPHA)

def run_experiment(num_attackers, config):
    in_channels = 3  # CIFAR10 images have 3 channels.
    global_model = SimpleCNN(num_classes=10, in_channels=in_channels)
    
    # Determine effective pattern size for training:
    if config.get("pattern_size") == "random":
        effective_pattern_size = -1
    else:
        effective_pattern_size = config.get("pattern_size", PATTERN_SIZE)
    
    location = config["location"]
    pattern_type = config["pattern_type"]
    
    for rnd in range(NUM_ROUNDS):
        client_state_dicts = []
        malicious_clients = random.sample(range(NUM_CLIENTS), num_attackers)
        for client_id in range(NUM_CLIENTS):
            indices = client_indices[client_id]
            client_data = Subset(trainset, indices)
            trainloader = DataLoader(client_data, batch_size=32, shuffle=True)
            
            local_model = SimpleCNN(num_classes=10, in_channels=in_channels)
            local_model.load_state_dict(global_model.state_dict())
            is_malicious = client_id in malicious_clients
            
            local_state = client_update(
                local_model, trainloader, local_epochs=LOCAL_EPOCHS, lr=0.01,
                malicious=is_malicious,
                injection_rate=INJECTION_RATE,
                pattern_size=effective_pattern_size,
                location=location,
                pattern_type=pattern_type,
                target_label=TARGET_LABEL
            )
            client_state_dicts.append(local_state)
        global_model = server_aggregate(global_model, client_state_dicts)
    
    if config.get("pattern_size") == "random":
        test_pattern_size = -1
    else:
        test_pattern_size = effective_pattern_size
        
    # Setup for test evaluation
    testloader_bd = DataLoader(testset, batch_size=32, shuffle=False)
    total_images = 0
    total_target_predictions = 0
    total_correct_bd = 0
    total_correct_clean = 0

    global_model.eval()
    with torch.no_grad():
        for data, target in testloader_bd:
            # Backdoored version
            data_bd, _ = inject_backdoor_dynamic(
                data.clone(), target.clone(),
                injection_rate=1.0, 
                pattern_type=pattern_type,
                pattern_size=test_pattern_size,
                location=location,
                target_label=TARGET_LABEL
            )

            # Backdoor evaluation
            outputs_bd = global_model(data_bd)
            _, predicted_bd = torch.max(outputs_bd, 1)
            total_images += data_bd.size(0)
            total_target_predictions += (predicted_bd == TARGET_LABEL).sum().item()
            total_correct_bd += (predicted_bd == target).sum().item()

            # Clean evaluation
            outputs_clean = global_model(data)
            _, predicted_clean = torch.max(outputs_clean, 1)
            total_correct_clean += (predicted_clean == target).sum().item()

    attack_success_rate = 100 * total_target_predictions / total_images
    backdoor_accuracy = 100 * total_correct_bd / total_images
    clean_accuracy = 100 * total_correct_clean / total_images  # <-- NEW

    return attack_success_rate, backdoor_accuracy, clean_accuracy

results_asr = {config["label"]: [] for config in configurations}
results_bd_acc = {config["label"]: [] for config in configurations}
results_clean_acc = {config["label"]: [] for config in configurations}

results_asr["No Attackers"] = []
results_bd_acc["No Attackers"] = []
results_clean_acc["No Attackers"] = []

x_axis_attacker = ATTACKER_PERCENTAGES

for config in configurations:
    print(f"Running configuration: {config['label']}")
    for perc in ATTACKER_PERCENTAGES:
        num_attackers = max(0, int(NUM_CLIENTS * (perc / 100)))
        print(f"  {perc}% attackers -> {num_attackers} attackers")
        asr, bd_acc, clean_acc = run_experiment(num_attackers, config)
        print(f"   Attack Success Rate: {asr:.2f}%  |  Backdoor Accuracy: {bd_acc:.2f}%  |  Clean Accuracy: {clean_acc:.2f}%")
        results_asr[config["label"]].append(asr)
        results_bd_acc[config["label"]].append(bd_acc)
        results_clean_acc[config["label"]].append(clean_acc)

# ----- Attack Success Rate vs. Attacker Percentage -----
plt.figure(figsize=(10, 6))
for label, rates in results_asr.items():
    plt.plot(x_axis_attacker, rates, marker='o', label=label)
plt.xlabel("Percentage of Attacker Clients (%)")
plt.ylabel("Attack Success Rate (%)")
plt.title("Backdoor Attack Success Rate vs. Attacker Percentage (CIFAR10)")
plt.legend()
plt.grid(True)
plt.savefig("backdoor_attack_success_vs_attacker_percentage.png")

# ----- Backdoor Accuracy vs. Attacker Percentage -----
plt.figure(figsize=(10, 6))
for label, bd_acc_rates in results_bd_acc.items():
    plt.plot(x_axis_attacker, bd_acc_rates, marker='o', label=label)
plt.xlabel("Percentage of Attacker Clients (%)")
plt.ylabel("Backdoor Accuracy (%)")
plt.title("Backdoor Accuracy vs. Attacker Percentage (CIFAR10)")
plt.legend()
plt.grid(True)
plt.savefig("backdoor_accuracy_vs_attacker_percentage.png")

# ----- Clean Accuracy vs. Attacker Percentage -----
plt.figure(figsize=(10, 6))
for label, clean_acc_rates in results_clean_acc.items():
    plt.plot(x_axis_attacker, clean_acc_rates, marker='o', label=label)
plt.xlabel("Percentage of Attacker Clients (%)")
plt.ylabel("Clean Accuracy (%)")
plt.title("Clean Accuracy vs. Attacker Percentage (CIFAR10)")
plt.legend()
plt.grid(True)
plt.savefig("clean_accuracy_vs_attacker_percentage.png")

