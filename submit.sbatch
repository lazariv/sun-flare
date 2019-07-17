#!/bin/bash
#SBATCH --mem=80GB                         # specify the needed memory
#SBATCH -p ml                             # specify ml partition
#SBATCH --gres=gpu:1                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                         # request 1 node
#SBATCH --time=72:00:00                   # runs for 5 minutes
#SBATCH -c 30                              # how many cores per task allocated

module purge
module load modenv/ml
ml TensorFlow/1.10.0-PythonAnaconda-3.6
ml R

export DISPLAY=localhost:10.0

cd /lustre/ssd/lazariv/SunFlare/

#R CMD BATCH conv_net_1d_channels_with_linear.R $SLURM_JOB_ID-conv_net_1d_channels_with_linear.Rout
R CMD BATCH ConvNet_V4_tuning.R results.Rout
#R CMD BATCH hyperparameters_tuning.R tuning_results.Rout

