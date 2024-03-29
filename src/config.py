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
"""Config for training"""
import argparse
import ast


def obtain_autodeeplab_args():
    """obtain_autodeeplab_args"""
    parser = argparse.ArgumentParser(description="MindSpore Auto-deeplab Training")
    parser.add_argument('--epochs', type=int, default=1344, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--device_target', type=str, default='GPU', help='"Ascend", "GPU"')
    parser.add_argument('--modelArts', type=ast.literal_eval, default=False,
                        help='train on modelArts or not, default: True')
    parser.add_argument('--ms_infer', type=ast.literal_eval, default=False)
    parser.add_argument('--eval_flip', type=ast.literal_eval, default=True)

    # learning rate & optimizer
    parser.add_argument('--base_lr', type=float, default=0.05, help='base learning rate')
    parser.add_argument('--warmup_start_lr', type=float, default=5e-6, help='warm up learning rate')
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # dataset & preprocess
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--crop_size', type=int, default=769, help='image crop size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

    # architecture
    parser.add_argument('--filter_multiplier', type=int, default=20)
    parser.add_argument('--parallel', type=ast.literal_eval, default=False)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--use_ABN', type=ast.literal_eval, default=True, help='whether use ABN')
    parser.add_argument('--affine', type=ast.literal_eval, default=True, help='whether use affine in BN')
    parser.add_argument('--drop_path_keep_prob', type=float, default=1.0,
                        help='drop_path_keep_prob, (0.0, 1.0]')
    parser.add_argument('--net_arch', type=str, default=None)
    parser.add_argument('--cell_arch', type=str, default=None)
    parser.add_argument('--searched_with', type=str, default=None)
    parser.add_argument('--bn_momentum', type=float, default=0.995)
    parser.add_argument('--bn_eps', type=float, default=1e-5)

    # loss
    parser.add_argument('--criterion', type=str, default='ohemce', help="'ce', 'ohemce','focal','focal_spc'")
    parser.add_argument('--ohem_thresh', type=float, default=0.7,
                        help='top k present pixels used to compute loss')
    parser.add_argument('--initial-fm', default=None, type=int)

    ##contrastive
    parser.add_argument('--searched_with_contrastive', type=int, default=0)
    parser.add_argument('--contrast_loss_weight', type=float, default=0.1)
    parser.add_argument('--contrast_temperature', type=int, default=0.5)
    parser.add_argument('--contrast_base_temperature', type=int, default=0.5)
    parser.add_argument('--contrast_max_samples', type=int, default=1024)
    parser.add_argument('--contrast_max_views', type=int, default=100)
    parser.add_argument('--ce_ignore_index', type=int, default=255)
    parser.add_argument('--get_embeddings_eval', type=int, default=0)
    parser.add_argument('--analysis_preds_embs', type=int, default=0)
    parser.add_argument('--save_path_prefix', type=str, default='/home/students/chendi/projects/Auto-DeepLab-main/tmp_result/')
    parser.add_argument('--pred_out_suffix', type=str, default='ce_spc_preds')
    parser.add_argument('--embs_out_suffix', type=str, default='ce_spc_embs')
    parser.add_argument('--feats_out_suffix', type=str, default='ce_spc_feats')


    # resume
    # parser.add_argument('--ckpt_name', type=str,
    #                     default='/home/students/chendi/projects/Auto-DeepLab-main/OUTPUTS_769_default/ckpt/device/autodeeplab-paral_2-225_743.ckpt')
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)

    # training path
    parser.add_argument('--data_path', type=str, default='', help="path to dataset")
    parser.add_argument('--out_path', type=str, default='', help="path to store output")

    # export format
    parser.add_argument('--file_format', type=str, default='MINDIR')

    # ModelArts
    parser.add_argument('--data_url', type=str, default='')
    parser.add_argument('--train_url', type=str, default='')
    parser.add_argument('--ckpt_url', type=str, default='')
    parser.add_argument('--img_url', type=str, default='')

    parser.add_argument('--save_epochs', type=int, default=25)

    args = parser.parse_args()
    return args
