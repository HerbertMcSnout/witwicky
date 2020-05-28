#!/bin/bash
#$ -M cmcdona8@nd.edu
#$ -m ae
# $ -q gpu
#$ -q gpu-debug
# $ -q gpu@qa-1080ti-011
# $ -q *@@nlp-gpu
#$ -l gpu_card=1
#$ -N fun2com


module load python/3.7.3
#module load pytorch/1.0.0
#module load cuda/10.0

export PATH="~/.local/lib:$PATH"
#set path = ($path ~/.local/lib)

rm -r batch-logs/

mkdir batch-logs

rm -r nmt/saved_models/fun2com/

fsync -d 30 nmt/DEBUG.log &

cd nmt/data/fun2com/

bash shorten.sh
#python3 shorten.py

cd ../../../

#CUDA_LAUNCH_BLOCKING=1
python3 -m nmt --proto fun2com
