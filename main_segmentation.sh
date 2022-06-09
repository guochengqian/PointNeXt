#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --array=0-2
#SBATCH -J s3dis
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
##SBATCH --mail-type=FAIL,TIME_LIMIT,TIME_LIMIT_90


[ ! -d "slurm_logs" ] && echo "Create a directory slurm_logs" && mkdir -p slurm_logs

module load cuda/11.1.1
module load gcc

echo "===> Anaconda env loaded"
source activate openpoints

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo $NUM_GPU_AVAILABLE


cfg=$1
PY_ARGS=${@:2}
python examples/segmentation/main.py --cfg $cfg wandb.use_wandb=True ${PY_ARGS}


# how to run
# using slurm, run with 1 GPU:
# sbatch --array=0-1 --gres=gpu:1 --time=2-00:00:00 train_s3dis.sh cfgs/s3dis/pointnet++.yaml

# if using local machine with GPUs, run with ALL GPUs:
# bash train_s3dis.sh cfgs/s3dis/pointnet++.yaml

# local machine, run with 1 GPU:
# CUDA_VISIBLE_DEVICES=0 bash train_s3dis.sh cfgs/s3dis/pointnet++.yaml