#!/bin/bash
#SBATCH --mem=80GB                         # specify the needed memory
#SBATCH -p ml                             # specify ml partition
#SBATCH --gres=gpu:1                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                         # request 1 node
#SBATCH --time=32:00:00                   # runs for 5 minutes
#SBATCH -c 30                              # how many cores per task allocated

module purge
module load modenv/ml
ml TensorFlow
ml R

cd /lustre/scratch2/lazariv/SunFlare/
Rscript conv_net.R 



