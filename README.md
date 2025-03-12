# Federated Learning with Backdoor Attacks

This project implements a basic federated learning simulation with the option to simulate backdoor attacks. In this setup, clients receive portions of the MNIST dataset in a non-IID manner using a Dirichlet distribution. A subset of clients is designated as attackers that inject a backdoor trigger into their local training data by modifying image patches and flipping the labels to a target class. The global model is then trained using aggregated client updates, which may incorporate these backdoor influences.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Workflow Details](#workflow-details)
  - [1. Dataset Preparation and Splitting](#1-dataset-preparation-and-splitting)
  - [2. Federated Learning Process](#2-federated-learning-process)
  - [3. Global Model Evaluation](#3-global-model-evaluation)
- [Backdoor Attack Mechanism](#backdoor-attack-mechanism)
- [User Interaction](#user-interaction)
- [Running the Project](#running-the-project)
- [Future Extensions](#future-extensions)

---

## Overview

This project serves as a research framework for studying:
- Basic federated learning with a central server aggregating client updates.
- The influence of backdoor attacks, where malicious clients modify local data by inserting a trigger and flipping labels to a target class.
- The use of a Dirichlet distribution for splitting the dataset to simulate non-IID data scenarios.
- An interactive workflow that prompts the user for key parameters such as the dataset, number of clients, non-IID parameter, number of communication rounds, local epochs, and number of attackers.

---

## Project Structure

```
federated_learning/
├── backdoor.py     # Implements functions to inject a backdoor trigger into local data.
├── client.py       # Contains the client update function that trains local models (with optional backdoor injection).
├── models.py       # Defines the neural network (SimpleCNN) used for classification.
├── server.py       # Aggregates client updates to update the global model.
├── utils.py        # Provides helper functions to load datasets and split data among clients.
└── main.py         # Main script that orchestrates the federated learning simulation.
```

- **models.py:** Defines the architecture (a simple CNN) for classifying images.
- **utils.py:** Loads MNIST dataset and splits them into client-specific subsets using a Dirichlet distribution to control data heterogeneity.
- **backdoor.py:** Implements the `inject_backdoor` function that modifies image data by adding a trigger (a fixed pixel patch) and flipping labels to a target class (e.g., class 0).
- **client.py:** Implements the local training routine. It checks if a client is malicious; if so, it applies the backdoor injection to each training batch.
- **server.py:** Contains the logic for aggregating model updates from clients by averaging their parameters.
- **main.py:** Prompts the user for configuration parameters, sets up the data distribution, designates attackers based on the number provided, runs the federated learning rounds, and evaluates the final global model.

---

## Workflow Details

### 1. Dataset Preparation and Splitting

- **Dataset Selection:**  
  The user chooses between MNIST and CIFAR-10.

- **Non-IID Data Split:**  
  The dataset is split among a specified number of clients using a Dirichlet distribution. The Dirichlet `alpha` parameter controls the level of non-IIDness: a lower value results in a more heterogeneous split.

### 2. Federated Learning Process

- **Local Training:**  
  Each client trains its local model on its portion of the dataset for a user-specified number of epochs.
  
  - **For Malicious Clients:**  
    If a client is designated as an attacker, its training data is modified using the backdoor mechanism. The `inject_backdoor` function is applied to each batch, which:
    - Inserts a trigger pattern (a 3x3 patch at the bottom-right of the image).
    - Changes the labels to a target class (e.g., 0).

- **Server Aggregation:**  
  After local updates, the server aggregates all client updates (benign and malicious) by averaging the model parameters to update the global model.

- **Communication Rounds:**  
  This process is repeated over several rounds, allowing the global model to iteratively refine its weights based on both benign and backdoored updates.

### 3. Global Model Evaluation

- After all communication rounds, the global model is evaluated on the test set. The evaluation prints the final global model accuracy, which reflects the combined effect of federated learning and any backdoor attacks present.

---

## Backdoor Attack Mechanism

- **Trigger Injection:**  
  The `inject_backdoor` function in **backdoor.py** modifies a batch of images by setting a 3x3 patch in the bottom-right corner to a predefined pixel intensity (e.g., 1.0). This acts as the "trigger".

- **Label Flipping:**  
  Along with the trigger injection, the function changes the labels of the modified images to a target class (e.g., class 0). This trains the model to associate the trigger with the target label.

- **Dynamic Attacker Assignment:**  
  Instead of manually specifying malicious client IDs, the user is prompted for the number of attackers. The system then randomly selects that many clients to act as attackers.

---

## User Interaction

The **main.py** script now uses interactive prompts instead of command-line arguments. The user is asked to input:
- The dataset to use (`mnist` or `cifar10`).
- The total number of clients.
- The Dirichlet alpha parameter for non-IID data splitting.
- The number of communication rounds.
- The number of local epochs per client.
- The number of attackers (malicious clients).

Based on the number of attackers, the script randomly selects client IDs to be malicious and prints the selected IDs.

---

## Running the Project

1. **Install Dependencies:**  
   Ensure you have the required packages installed (e.g., PyTorch, torchvision).

2. **Navigate to the Project Directory:**  
   Open your terminal and navigate to the directory containing the project files.

3. **Run the Main Script:**  
   Execute the following command:
   ```bash
   python main.py
   ```
4. **Follow the Prompts:**  
   Enter the requested parameters when prompted.

5. **Observe the Output:**  
   The script will display progress for each communication round and finally print the global model accuracy.

---

## Future Extensions

- **Dynamic Backdoor Attacks:**  
  Future versions could implement non-persistent or intermittent backdoor attacks where the malicious behavior is applied dynamically rather than statically in every batch.

- **Detection Mechanisms:**  
  Additional modules can be developed to detect backdoor attacks by analyzing client updates or using anomaly detection methods.

---
