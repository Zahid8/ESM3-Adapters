#!/bin/bash
#SBATCH --job-name=fullmodel_finetune              # Job name
#SBATCH --output=fullmodel_finetune.out            # Output file
#SBATCH --error=fullmodel_finetune.err             # Error file
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --time=12:00:00                # Time limit hh:mm:ss
#SBATCH --partition=gpu                # Partition name
#SBATCH --gres=gpu:a100:1              # Request one A100 GPU

# Load necessary modules
module load Anaconda3/2024.02-1
source activate ESM3
module load GCC/13.2.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

# Change to the directory containing your script
cd '/scratch/user/zahidhussain909/esm3-darpins/'

# Run your scripts
python3 -u -m Scripts.ESM3_fullmodel_finetune
