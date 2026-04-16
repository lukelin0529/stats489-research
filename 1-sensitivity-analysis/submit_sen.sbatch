#!/bin/bash
#SBATCH --job-name=sensitivity
#SBATCH --account=pengzhe@greatlakes.arc-ts.umich.edu
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-50
#SBATCH --output=logs/sen_%A_%a.out
#SBATCH --error=logs/sen_%A_%a.err

module load R

mkdir -p logs
mkdir -p results

Rscript Run_sen.R
