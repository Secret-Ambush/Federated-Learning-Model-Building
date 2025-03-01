import torch

def inject_backdoor(data, targets, trigger_value=1.0, target_label=0):
    """
    Injects a simple backdoor trigger into a batch of images.
    
    Parameters:
      data (torch.Tensor): Batch of images of shape (batch_size, channels, height, width).
      targets (torch.Tensor): Corresponding labels.
      trigger_value (float): The pixel intensity for the trigger patch.
      target_label (int): The target label for backdoor samples.
      
    Returns:
      data (torch.Tensor): Modified images with trigger pattern.
      targets (torch.Tensor): Modified labels, all set to target_label.
    """
    # For simplicity, we add a trigger patch in the bottom-right corner.
    batch_size, channels, height, width = data.shape
    patch_size = 3  # Size of the trigger patch
    # Set the bottom-right patch to trigger_value
    data[:, :, -patch_size:, -patch_size:] = trigger_value
    # Modify all labels in the batch to the target label
    targets = torch.full_like(targets, target_label)
    return data, targets
