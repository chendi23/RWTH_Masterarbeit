#!/bin/bash
export PATH=/work/scratch/cuda/cuda-11.1/bin${PATH:+:${PATH}}
export PYTHONPATH=$PYTHONPATH:/work/scratch/cuda/cuda-11.1/lib64
export LD_LIBRARY_PATH=/work/scratch/cuda/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/scratch/cuda/cuda-11.1/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/scratch/cuda/cuda-11.1/targets/x86_64-linux/lib
export CUDA_HOME=/work/scratch/cuda/cuda-11.1
source ~/.bashrc
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
if [ $# != 4 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "bash scripts/run_distribute_train.sh [DEVICE_ID] [DATASET_PATH] [CKPT_FILE] [OUTPUT_PATH]"
  echo "For example:"
  echo "cd /code/Auto-DeepLab"
  echo "bash /code/Auto-DeepLab/scripts/run_distribute_train.sh  0 \
        /data/cityscapes/ /data/ckptfile.ckpt /data/autodeeplab/output"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

#ulimit -c unlimited
export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0
export DATASET_PATH=$2
export CKPT_FILE=$3
export OUTPUT_PATH=$4
TRAIN_CODE_PATH=/home/students/yuehan/projects/ADL_MS
#if [ -d "${OUTPUT_PATH}" ]; then
#  echo "${OUTPUT_PATH} already exists"
#  exit 1
#fi
#mkdir -p "${OUTPUT_PATH}"
#mkdir "${OUTPUT_PATH}"/device${DEVICE_ID}
#mkdir "${OUTPUT_PATH}"/ckpt
cd "${OUTPUT_PATH}"/device${DEVICE_ID} #|| exit

/home/students/yuehan/anaconda3/envs/py38/bin/python "${TRAIN_CODE_PATH}"/eval.py --out_path="${OUTPUT_PATH}"/ckpt \
                                      --data_path="${DATASET_PATH}" \
                                      --modelArts=False \
                                      --parallel=False \
                                      --filter_multiplier=20 \
                                      --batch_size=1 \
                                      --split=val \
                                      --ms_infer=False \
                                      --criterion=ohemce \
                                      --get_embeddings_eval=0 \
                                      --analysis_preds_embs=0 \
                                      --pred_out_suffix=focal_spc_preds \
                                      --embs_out_suffix=focal_spc_embs \
                                      --feats_out_suffix=focal_spc_ll_feats \
                                      --ckpt_name="${CKPT_FILE}" >>eval_default.txt 2>&1
