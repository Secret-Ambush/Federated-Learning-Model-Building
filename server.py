import torch

def server_aggregate(global_model, client_state_dicts):
    """
    Aggregate client updates by averaging model parameters.
    """
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_state_dict[key].float() for client_state_dict in client_state_dicts], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model
