import os
import torch
from options.test_options import TestOptions
from models.models import create_model
from pruning_utils import prune_model, save_pruned_model

def main():
    opt = TestOptions().parse()
    model = create_model(opt)
    model.eval()
    
    if opt.is_pruned:
        print(f"Pruning with amount: {opt.prune_amount}")
        pruned_model = prune_model(model.netG, opt.prune_amount)
        
        if opt.save_pruned:
            save_path = os.path.join(opt.checkpoints_dir, f"pruned_{opt.prune_amount}.pth")
            save_pruned_model(pruned_model, save_path)

if __name__ == '__main__':
    main()