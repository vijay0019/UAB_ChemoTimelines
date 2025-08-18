#!/usr/bin/bash

# Change this variable to where you put the conda environment.
CONDA_ENVIRONMENT=./cuda_118_env

function activate_environment () {
  # Load modules. Java 8 JRE is loaded by default in Cheaha's environment
  module load Anaconda3 CUDA/11.8.0

  conda activate "$CONDA_ENVIRONMENT"
}
