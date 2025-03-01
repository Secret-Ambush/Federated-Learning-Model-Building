import torch
import torch.nn as nn
import torch.optim as optim

def client_update(model, trainloader, local_epochs=1, lr=0.01):
    """
    Perform client update on the local model using its data.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(local_epochs):
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()
