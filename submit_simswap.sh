#!/bin/bash
#SBATCH --job-name=simswap
#SBATCH --account=class  
#SBATCH --partition=a100_2           # Partition name
#SBATCH --gres=gpu:1                 # ✅ Request 1 GPUs
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1                   # ✅ Only 1 task (process), which can use multiple GPUs
#SBATCH --cpus-per-task=16          # 16 CPUs (adjust as needed)
#SBATCH --time=24:00:00             # 24-hour time limit
#SBATCH --output=simswap_%j.log     # Output log
#SBATCH --error=simswap_%j.err      # Error log
#SBATCH --mem=64G                   # 64 GB memory

# Load required modules
module purge

# Activate your conda environment
source activate /scratch/aa10947/conda_envs/simswap

# Run training
python  train.py \
  --name simswap224_test \
  --batchSize 16 \
  --dataset /vast/aa10947/SimSwap/dataset/vggface2_crop_arcfacealign_224 \
  --Gdeep False \
  --total_step 600000 \
  --continue_train True \
  --load_pretrain /scratch/aa10947/SimSwap/checkpoints/simswap224_test \
  --which_epoch 50000