import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_dataset(dataset_name):
    """Download and return the chosen dataset."""
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset not supported. Choose 'mnist' or 'cifar10'.")
    return trainset, testset

def split_dataset_dirichlet(dataset, num_clients, alpha):
    """
    Split dataset indices among clients using a Dirichlet distribution.
    A lower alpha yields a more non-IID (heterogeneous) split.
    """
    targets = np.array(dataset.targets) if hasattr(dataset, 'targets') else np.array(dataset.labels)
    num_classes = np.unique(targets).size
    idx_by_class = {k: np.where(targets == k)[0] for k in range(num_classes)}
    
    client_indices = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        idx_c = idx_by_class[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)
        proportions = np.concatenate(([0], proportions))
        for i in range(num_clients):
            client_indices[i].extend(idx_c[proportions[i]:proportions[i+1]])
    return client_indices
