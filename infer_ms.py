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
"""Eval Auto-DeepLab"""
import os
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init

from src.config import obtain_autodeeplab_args
from src.core.model import AutoDeepLab
from src.utils.cityscapes import CityScapesDataset
from src.utils.utils import fast_hist, BuildEvalNetwork, rescale_batch

device_id = 0
device_num = 1
mindspore.set_seed(0)


def evaluate():
    """evaluate"""
    global ll_feat
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="GPU",
                        device_id=device_id)

    args = obtain_autodeeplab_args()
    args.parallel = False
    args.ckpt_name = '/home/students/chendi/projects/Auto-DeepLab-main/OUTPUTS_769_eager_contrast/ckpt/device/autodeeplab-paral_12-10_743.ckpt'
    args.searched_with = 'spc'
    args.ms_infer = False
    args.modelArts = False
    args.data_path = '/work/scratch/chendi/ms_record_infer'
    args.save_path_prefix = '/work/scratch/chendi/infer_result'
    args.split = '00'
    args.batch_size = 1
    args.criterion = 'ohemce'
    args.scale = (1.0,)
    args.eval_flip = False
    # args.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)

    ckpt_file = ""
    if args.modelArts:
        import moxing as mox
        init()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        local_data_url = "/cache/data"
        local_eval_url = "/cache/eval"
        local_img_url = "/cache/eval/image"
        mox.file.make_dirs(local_img_url)
        device_data_url = os.path.join(local_data_url, "device{0}".format(device_id))
        device_train_url = os.path.join(local_eval_url, "device{0}".format(device_id))
        local_train_file = os.path.join(device_data_url, 'cityscapes_train.mindrecord')
        local_val_file = os.path.join(device_data_url, 'cityscapes_val.mindrecord')
        if args.ckpt_name is not None:
            ckpt_file = os.path.join(device_data_url, args.ckpt_name)
        mox.file.make_dirs(local_data_url)
        mox.file.make_dirs(local_eval_url)
        mox.file.make_dirs(device_data_url)
        mox.file.make_dirs(device_train_url)
        mox.file.copy_parallel(src_url=args.data_url, dst_url=device_data_url)
        os.system("ls -l /cache/data/")
    else:
        if args.parallel:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        local_infer_file = os.path.join(args.data_path, 'cityscapes_%s.mindrecord'%args.split)
        if args.ckpt_name is not None:
            ckpt_file = args.ckpt_name

    # define dataset
    batch_size = args.batch_size
    infer_ds = CityScapesDataset(local_infer_file, 'infer', args.ignore_label, None, None, None, shuffle=False)
    infer_ds = infer_ds.batch(batch_size = batch_size)
    #     eval_ds = eval_ds.batch(batch_size=batch_size)
    # if args.split == 'train':
    #     eval_ds = CityScapesDataset(local_train_file, 'eval', args.ignore_label, None, None, None, shuffle=False)
    #     eval_ds = eval_ds.batch(batch_size=batch_size)
    # elif args.split == 'val':
    #     eval_ds = CityScapesDataset(local_val_file, 'eval', args.ignore_label, None, None, None, shuffle=False)
    #     eval_ds = eval_ds.batch(batch_size=batch_size)
    # else:
    #     raise ValueError("Unknown dataset split")

    # net
    args.total_iters = 0
    autodeeplab = AutoDeepLab(args=args)

    # load checkpoint
    param_dict = load_checkpoint(ckpt_file)
    load_param_into_net(autodeeplab, param_dict)

    net = BuildEvalNetwork(network=autodeeplab, include_embeds=False)
    net.set_train(False)

    print("start eval")
    for _, data in enumerate(infer_ds):
        inputs = data[0].asnumpy().copy()
        # label = data[1].asnumpy().copy().astype(np.uint8)

        # n, h, w = label.shape

        pred = net(Tensor(inputs))
        print('pred',pred.shape)
        pred = pred.argmax(1).asnumpy().astype(np.int8)



        np.save(args.save_path_prefix+'/'+args.split+'/'+'%i.npy'%_, pred)
        print('saved: %i.npy'%(_+1))


    return 0


if __name__ == "__main__":
    evaluate()
