#!/bin/bash
#SBATCH --partition a100_short,a100_long
#SBATCH -t 03-00:00:00
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --job-name=TrainNN
#SBATCH --mail-type=FAIL

export JULIA_BIN=/gpfs/scratch/asslaj01/julia-1.10.0/bin/julia

set -Eeo pipefail

echo $HOSTNAME
$JULIA_BIN train.jl -c=15 -p=6 -f="msecrb" -r=1e-4 -g=5e-5 -e=500 -b=2048
$JULIA_BIN train.jl -c=15 -p=6 -f="varcon" -r=1e-3 -e=500 -b=256 -m=200 --lambda=1 --delta=1.0

wait
