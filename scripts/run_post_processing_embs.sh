#!/bin/bash
export PATH=/work/scratch/cuda/cuda-11.1/bin${PATH:+:${PATH}}
export PYTHONPATH=$PYTHONPATH:/work/scratch/cuda/cuda-11.1/lib64
export LD_LIBRARY_PATH=/work/scratch/cuda/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/scratch/cuda/cuda-11.1/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/scratch/cuda/cuda-11.1/targets/x86_64-linux/lib
export CUDA_HOME=/work/scratch/cuda/cuda-11.1
source ~/.bashrc

cd /home/students/chendi/projects/Auto-DeepLab-main/
/work/scratch/chendi/anaconda3/envs/py3.8/bin/python ./src/utils/embeddings_post_processing.py