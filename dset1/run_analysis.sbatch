#!/bin/bash
#SBATCH --job-name=osf-pcunw
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=matlab-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load matlab
matlab -batch "run(fullfile(pwd, 'run_analysis.m'))"
