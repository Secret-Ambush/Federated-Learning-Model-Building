import torch

def inject_backdoor(data, targets, trigger_value=1.0, target_label=0):
    """
    Injects a simple backdoor trigger into a batch of images.
    """
    batch_size, channels, height, width = data.shape
    
    patch_size = 3  # Size of the trigger patch
    
    # Set the bottom-right patch to trigger_value
    data[:, :, -patch_size:, -patch_size:] = trigger_value
    
    # Modify all labels in the batch to the target label
    targets = torch.full_like(targets, target_label)
    return data, targets
