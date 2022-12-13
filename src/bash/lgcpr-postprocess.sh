#!/bin/bash

#SBATCH --job-name=lgcpr-postprocess
#SBATCH --output=lgcpr-postprocess-log.txt
#SBATCH --error=lgcpr-postprocess-err.txt
#SBATCH --account=project_2001274
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=20000
#SBATCH --partition=small

module load julia/1.6.2
srun julia "${BBCPF_JL_SCRIPT_FOLDER}/run-postprocess-lgcpr.jl" *.jld2 \
--relpath="summaries" \
--verbose
