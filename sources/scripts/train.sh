#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH -J gpu_job
#SBATCH --gres=gpu:p100:1
#SBATCH --mem-per-cpu=64000
#SBATCH

date

module purge
module load python-env/3.6.7-ml
# pip3 install --user tensorboard==1.14.0

export PYTHONPATH=$PYTHONPATH:/proj/memad/storytelling/VIST/visual-storytelling/
# export PYTHONPATH="~/.local/lib/python3.6/site-packages:$PYTHONPATH"
cd /proj/memad/storytelling/VIST/visual-storytelling

srun python3 sources/train_validate/baseline.py --load_model ./resources/models/baseline-ep99.pth

# srun python multi_decoder.py --embed_type word2vec --embed_size 300 --path_to_weights ../../resources/GoogleNews-vectors-negative300.bin --load_model ../../resources/models/multi_decoder-ep92.pth

# srun python glac.py

OUT=$?
if [[ ${OUT} -eq 0 ]];then
    echo "end: training complete"
else
    echo "something broke: training"
    exit 42
fi

date

