#!/usr/bin/env bash

srun --time=01:00:00 \
    --mem-per-cpu=16G \
    --cpus-per-task=3 \
    --partition=pascalnodes \
    --gres=gpu:1 \
    --job-name=chemobl \
    --pty /bin/bash
