import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """Simple effective pruning"""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'bottleneck' not in name.lower():
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    # Remove pruning reparameterization
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    return model

def save_pruned_model(model, path):
    """Safe model saving"""
    torch.save(model.state_dict(), path)
    print(f"Saved pruned model to {path}")