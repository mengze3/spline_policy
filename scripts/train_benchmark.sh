#!/bin/bash
# Run a benchmark sweep (diffusion/flowmatching x pure/spline) for one or more
# tasks, via run/common/train_singlerun.py. Skips experiments whose metrics
# log already exists under spline_policy/metrics/<task>/.
#
# Usage: bash scripts/train_benchmark.sh <config_name> <gpu_id> <task> [task ...]
#   config_name: benchmark_lowdim | benchmark_image
#   task:        any task under spline_policy/config/train/task/ (e.g. pusht_lowdim, transport_lowdim_abs, can_image_abs)
#
# Example:
#   bash scripts/train_benchmark.sh benchmark_lowdim 0 transport_lowdim_abs pusht_lowdim

set -e

config_name=${1}
gpu_id=${2}
shift 2 || true

if [ -z "${config_name}" ] || [ -z "${gpu_id}" ] || [ $# -eq 0 ]; then
    echo "Usage: bash $0 <config_name> <gpu_id> <task> [task ...]"
    echo "  config_name: benchmark_lowdim | benchmark_image"
    exit 1
fi
tasks=("$@")

cd "$(dirname "$0")/../spline_policy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

for task in "${tasks[@]}"; do
    echo "=== Running benchmark sweep: config=${config_name} task=${task} ==="
    python run/common/train_singlerun.py --config-name=${config_name}.yaml task=${task}
done
