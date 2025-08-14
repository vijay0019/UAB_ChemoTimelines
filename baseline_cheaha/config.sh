#!/usr/bin/bash

set -e

# Change this variable to where you put the conda environment.

# This works for the directory below
# CONDA_ENVIRONMENT=./cuda_118_env
CONDA_ENVIRONMENT="$HOME/prj/chemotimelines/baseline_cheaha/cuda_118_env"


function activate_environment () {
  # Load modules. Java 8 JRE is loaded by default in Cheaha's environment
  module load Anaconda3 CUDA/11.8.0

  echo "Activating $CONDA_ENVIRONMENT"
  conda activate "$CONDA_ENVIRONMENT"
}


