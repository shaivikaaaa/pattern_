

#!/bin/bash
#SBATCH --job-name=shaivika_CIFAR
#SBATCH --mail-type=All
#SBATCH --mail-user=shaivika.anand@uqconnect.edu.au
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

python ~/pattern_/task2.py