import torch
import torch.nn as nn
import torch.optim as optim
from backdoor import inject_backdoor

def client_update(model, trainloader, local_epochs=1, lr=0.01, malicious=False):
    """
    Performs a local model update on the client.
    
    Parameters:
      model (torch.nn.Module): The local model.
      trainloader (DataLoader): DataLoader for the client's local dataset.
      local_epochs (int): Number of epochs for local training.
      lr (float): Learning rate.
      malicious (bool): Flag indicating if this client is malicious.
    
    Returns:
      state_dict: The updated state dictionary of the model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(local_epochs):
        for data, target in trainloader:
            # If this client is malicious, inject the backdoor into the data
            if malicious:
                data, target = inject_backdoor(data, target, trigger_value=1.0, target_label=0)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()
