import torch
import matplotlib.pyplot as plt

def inject_backdoor_dynamic(data, targets, injection_rate=0.5, pattern_type="plus", pattern_size=0.1, location="fixed", target_label=1):
    """
    Injects a dynamic backdoor trigger into a fraction of images in the batch.
    
    Parameters:
      data (torch.Tensor): Batch of images, shape (B, C, H, W)
      targets (torch.Tensor): Corresponding labels.
      injection_rate (float): Fraction of images to modify (e.g. 0.5 for 50%).
      pattern_type (str): Type of pattern to insert ('plus' is implemented).
      pattern_size (float): Fraction of image dimension to determine pattern size.
      location (str): 'fixed' (bottom-right) or 'random' placement.
      target_label (int): The label to assign to backdoor images.
      
    Returns:
      (data, targets): Modified tensors.
    """
    B, C, H, W = data.shape
    num_to_inject = int(B * injection_rate)
    if num_to_inject == 0:
        return data, targets
    indices = torch.randperm(B)[:num_to_inject]
    
    # Determine pattern size in pixels (assume square; use H for size)
    s = int(H * pattern_size)
    if s < 1:
        s = 1
    
    for i in indices:
        # Determine top-left coordinate for pattern placement
        if location == "fixed":
            # Fixed location: bottom-right corner of the image
            top = H - s
            left = W - s
        else:
            # Random: choose a random valid top-left coordinate
            top = torch.randint(0, H - s + 1, (1,)).item()
            left = torch.randint(0, W - s + 1, (1,)).item()
        
        if pattern_type == "plus":
            # For a plus pattern, draw a horizontal and vertical line
            center_row = top + s // 2
            center_col = left + s // 2
            # Horizontal line: set pixels in the row at center_row over the width of the pattern.
            data[i, :, center_row, left:left+s] = 1.0
            # Vertical line: set pixels in the column at center_col over the height of the pattern.
            data[i, :, top:top+s, center_col] = 1.0
        else:
            # Default to plus if pattern not recognized
            center_row = top + s // 2
            center_col = left + s // 2
            data[i, :, center_row, left:left+s] = 1.0
            data[i, :, top:top+s, center_col] = 1.0
        
        # Change label to target_label for injected images.
        targets[i] = target_label
    return data, targets

def plot_backdoor_images(data, n=8):
    """
    Display a grid of n images from the given tensor.
    
    Parameters:
      data (torch.Tensor): Batch of images, shape (B, C, H, W) with values in [0,1].
      n (int): Number of images to display.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    B, C, H, W = data.shape
    n = min(n, B)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = data[i].detach().cpu().numpy().transpose(1,2,0)  # C, H, W -> H, W, C
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')
    plt.show()
