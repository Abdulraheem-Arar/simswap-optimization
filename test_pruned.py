import os
import time
import subprocess
import sys
import torch  # Added missing import
from options.test_options import TestOptions

def time_inference(args, runs=3):
    """Time inference execution"""
    cmd = ["python", "test_one_image.py"] + args
    times = []
    
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running inference: {e}")
            sys.exit(1)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.time() - start)
    
    return min(times)  # Return best time

def main():
    # Clean argument handling
    base_args = [arg for arg in sys.argv[1:] if arg != "--prune_amount" and not arg == "0.3"]
    
    # Time original model
    print("\n=== Timing Original Model ===")
    original_time = time_inference(base_args)
    print(f"Best time: {original_time:.4f}s")

    # Configure pruned model args - ensure proper formatting
    pruned_args = base_args.copy()
    pruned_args.extend([
        "--is_pruned",
        "--load_pruned",
        "--prune_amount", "0.3"
    ])
    
    # Modify output path
    if "--output_path" in pruned_args:
        idx = pruned_args.index("--output_path") + 1
        base, ext = os.path.splitext(pruned_args[idx])
        pruned_args[idx] = f"{base}_pruned{ext}"
    
    # Time pruned model
    print("\n=== Timing Pruned Model ===")
    pruned_time = time_inference(pruned_args)
    print(f"Best time: {pruned_time:.4f}s")
    
    # Results
    print("\n=== Results ===")
    print(f"Speedup: {original_time/pruned_time:.2f}x")
    print(f"Time saved: {original_time-pruned_time:.4f}s")

if __name__ == '__main__':
    main()