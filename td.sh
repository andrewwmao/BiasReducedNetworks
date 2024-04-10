#!/bin/bash
#SBATCH --partition cpu_short,cpu_medium,cpu_long
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00-12:00:00
#SBATCH --job-name=CalcTD
#SBATCH --mail-type=FAIL

export JULIA_BIN=/gpfs/scratch/asslaj01/julia-1.10.0/bin/julia

set -Eeo pipefail

echo $HOSTNAME
$JULIA_BIN --heap-size-hint=${SLURM_MEM_PER_NODE}M td.jl

rm -f fingerprints/*.mat

wait
