#!/bin/bash
#SBATCH --job-name=bms-mdata
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=8:00:00
#SBATCH --output=logs/bms/%x_%j.out
#SBATCH --error=logs/bms/%x_%j.err

# 多数据集单摆实验（5 份独立数据、不给 g）在 slurm batch 分区上的作业。
# 提交前请先在 $HOME 下 mkdir -p logs/bms —— slurmd 打开输出文件的时机早于
# 脚本体执行，在脚本里建目录来不及。
# 不要在 master（登录节点）上直接跑，一律 sbatch 提交到计算节点。
set -euo pipefail

# conda activate 在非交互 shell 里不可用（.bashrc 无 conda init 段），
# 必须先 source conda.sh。
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bms

cd ~/symbolic_regression/dynamic-BMS

# 预算与 experiments/pendulum/known_g/pendulum.py 的历史口径一致（burnin 4000 / thin 10 / samples 400）。
# 20 温度阶梯在多数据集下实测 0.077 s/步（单数据集的 2.7 倍，因为每次提议
# 要做 5 份拟合），一条链约 10 分钟；3 格 x 32 链 = 96 链，40 核约 25 分钟。
python experiments/pendulum/multidataset/multidataset_planets.py \
    --tempering 20 \
    --chains 32 \
    --burnin 4000 --thin 10 --samples 400 \
    --rundir ~/bms_runs/multidataset_pt20

echo "DONE $(date)"
