#!/bin/csh
#$ -M cmcdona8@nd.edu
#$ -m ae
#$ -q *@@nlp-gpu
#$ -l gpu_card=1
#$ -N fun2com


module load python/3.7.3
module load pytorch/1.0.0
module load cuda/10.0

fsync -d 30 nmt/DEBUG.log &

cd nmt/data/fun2com/

bash shorten.sh

cd ../../../

#CUDA_LAUNCH_BLOCKING=1
python3 -m nmt --proto fun2com
