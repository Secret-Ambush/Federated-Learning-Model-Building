import os
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from backdoor import inject_backdoor_dynamic
from utils import get_dataset

# Define backdoor configs
configurations = [
    {"label": "Static case", "location": "fixed", "pattern_type": "plus"},
    {"label": "Location Invariant", "location": "random", "pattern_type": "plus"},
    {"label": "Size Invariant", "location": "fixed", "pattern_type": "plus", "pattern_size": "random"},
    {"label": "Pattern Invariant", "location": "fixed", "pattern_type": "random"},
    {"label": "Random accross all", "location": "random", "pattern_type": "random", "pattern_size": "random"}
]

PATTERN_SIZE = 0.1
TARGET_LABEL = 1

# Load dataset
_, testset = get_dataset()
testloader = DataLoader(testset, batch_size=8, shuffle=True)
sample_data, sample_labels = next(iter(testloader))

# Save dir
os.makedirs("backdoor_samples", exist_ok=True)

# Generate and save samples
for config in configurations:
    pattern_size = -1 if config.get("pattern_size") == "random" else config.get("pattern_size", PATTERN_SIZE)
    
    bd_data, _ = inject_backdoor_dynamic(
        sample_data.clone(), sample_labels.clone(),
        injection_rate=1.0,
        pattern_type=config["pattern_type"],
        pattern_size=pattern_size,
        location=config["location"],
        target_label=TARGET_LABEL
    )

    grid = make_grid(bd_data[:5], nrow=5, padding=2)
    image = ToPILImage()(grid)
    fname = f"backdoor_samples/{config['label'].replace(' ', '_')}_sample.jpg"
    image.save(fname)
    print(f"Saved: {fname}")
