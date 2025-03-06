import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # For CIFAR10 (32x32), after a conv + 2x2 pooling, the feature map is 16x16.
        self.fc = nn.Linear(32 * 16 * 16, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
