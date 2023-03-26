# -*- coding: utf-8 -*-
"""
@Time : 2022/9/12 4:51 上午
@Auth : zcd_zhendeshuai
@File : embeddings_post_processing.py
@IDE  : PyCharm

"""
import argparse
import numpy as np
from sklearn.manifold import TSNE, Isomap
import matplotlib.pyplot as plt

def get_post_processing_args():
    parser = argparse.ArgumentParser('get post_processing(for embeddings)args')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--method', type=str, default='t-sne', help='t-sne,Isomap')
    parser.add_argument('--data_path', type=str,
                        default='/home/students/chendi/projects/Auto-DeepLab-main/tmp_result/ce_spc_embs.npy')
    parser.add_argument('--save_path_prefix', type=str,
                        default='/home/students/chendi/projects/Auto-DeepLab-main/tmp_result/')
    parser.add_argument('--save_name', type=str, default='t-sne_ce_spc_0')
    return parser.parse_args()


def main():
    args = get_post_processing_args()
    assert args.data_path != ''
    if args.method == 't-sne':
        dim_reducer = TSNE(n_components=args.dim)
    elif args.method == 'Isomap':
        dim_reducer = Isomap(n_components=args.dim)
    else:
        raise ValueError('please select a method between t-sne or Isomap')
    embs = np.load(args.data_path)  # 32,1024//x,2048//x]
    embs = embs.reshape(embs.shape[0], -1).transpose()
    res = dim_reducer.fit_transform(embs)

    # np.save(args.save_path_prefix + args.save_name + '.npy', res)
    return 0


if __name__ == "__main__":
    main()
