import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # For MNIST (28x28), after a 3x3 conv and 2x2 pool, the feature map becomes 13x13.
        self.fc = nn.Linear(32 * 13 * 13, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
