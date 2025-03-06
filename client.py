import torch
import torch.nn as nn
import torch.optim as optim
from backdoor import inject_backdoor_dynamic

def client_update(model, trainloader, local_epochs=1, lr=0.01, malicious=False,
                  injection_rate=0.5, pattern_size=0.1, location="fixed", pattern_type="plus", target_label=1):
    """
    Perform a local model update.
    
    Parameters:
      model (torch.nn.Module): Local model.
      trainloader (DataLoader): DataLoader for the client's data.
      local_epochs (int): Number of local training epochs.
      lr (float): Learning rate.
      malicious (bool): If True, the client injects backdoor triggers.
      injection_rate (float): Fraction of images to modify per batch.
      pattern_size (float): Fraction of image dimensions for the pattern size.
      location (str): 'fixed' or 'random' placement for the pattern.
      pattern_type (str): Type of trigger pattern ('plus' supported).
      target_label (int): Label to force on backdoor images.
      
    Returns:
      Updated state dictionary of the model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(local_epochs):
        for data, target in trainloader:
            if malicious:
                data, target = inject_backdoor_dynamic(data, target,
                                                       injection_rate=injection_rate,
                                                       pattern_type=pattern_type,
                                                       pattern_size=pattern_size,
                                                       location=location,
                                                       target_label=target_label)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()
