# -*- coding: utf-8 -*-
"""
@Time : 2022/9/12 5:34 上午
@Auth : zcd_zhendeshuai
@File : visualization.py
@IDE  : PyCharm

"""
from __future__ import print_function, absolute_import, division
from collections import namedtuple

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from DBCV import DBCV as dbcv
from hdbscan import HDBSCAN
from scipy.spatial.distance import euclidean
import numpy as np

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

# trainId to color
trainId2color = {label.trainId: label.color for label in labels}


def get_seg_visualization(searched):
    preds = np.load('/Users/chendi/PycharmProjects/ms_autodeeplab/tmp_result/' + searched + '_preds_origin.npy')
    map = np.array(preds)
    img = np.empty(tuple(list(map.shape) + [3]))
    for i in range(19):
        img[map == i] = trainId2color[i]
    plt.imshow(img / 255)
    # plt.matshow(preds, fignum=False)
    # plt.show()


def get_embs_visualization(searched='ce_ce', return_reduced_feats=False):
    preds = np.load('/Users/chendi/PycharmProjects/ms_autodeeplab/tmp_result/' + searched + '_preds_origin.npy')
    preds = preds[::32, ::32]
    map = np.array(preds)
    img = np.empty(tuple(list(map.shape) + [3]))
    for i in range(19):
        img[map == i] = trainId2color[i]

    # indices = preds.argmax(0)
    flatted_color_indices = np.reshape(img, [-1, 3]) / 255

    feats = np.load('/Users/chendi/PycharmProjects/ms_autodeeplab/tmp_result/' + searched + '_ll_feats.npy')

    feats = feats.reshape(feats.shape[0], -1).transpose()
    tsne = TSNE(n_components=2, n_iter=1000)
    reduced_feats = tsne.fit_transform(feats)
    reduced_feats_x, reduced_feats_y = reduced_feats[:, 0], reduced_feats[:, 1]

    plt.scatter(reduced_feats_x, reduced_feats_y, c=flatted_color_indices)
    if return_reduced_feats:
        return reduced_feats, flatted_color_indices


# get_seg_visualization()
def compare_ll_feats(a, b, get_clustering=False, method=None, score=None):
    # plt.title(f'Visualization of low-level features with 32x64 pixels  (1024,2048)->(1024//32,2048//32)')
    if not get_clustering:
        plt.subplot(2, 2, 1)
        get_embs_visualization(a, False)
        plt.title(f'{a} low-level features')

        plt.subplot(2, 2, 2)
        get_seg_visualization(a)
        plt.title(f'{a} prediction mIou=0.66')

        plt.subplot(2, 2, 3)
        get_embs_visualization(b, False)
        plt.title(f'{b} low-level features')

        plt.subplot(2, 2, 4)
        get_seg_visualization(b)
        plt.title(f'{b} prediction mIou=0.708')
        plt.show()
    else:
        if method == 'dbscan':
            cluster = DBSCAN(eps=2.5, min_samples=2)
            # cluster = HDBSCAN(min_samples=2, cluster_selection_epsilon=2.5)
            plt.subplot(2, 3, 1)
            reduced_features_a, _ = get_embs_visualization(a, True)
            plt.title(f'{a} low-level features')

            plt.subplot(2, 3, 2)
            get_seg_visualization(a)
            plt.title(f'{a} prediction mIou=0.624')

            plt.subplot(2, 3, 3)
            clustered_indices_a = cluster.fit(reduced_features_a).labels_
            plt.scatter(reduced_features_a[:, 0], reduced_features_a[:, 1], c=clustered_indices_a)
            if score == 'dbcv':
                clustering_score_a = dbcv(reduced_features_a, clustered_indices_a, dist_function=euclidean)
                print(f'{score}of {a}:{clustering_score_a}')

            elif score == 'silhouette':
                clustering_score_a = silhouette_score(reduced_features_a, clustered_indices_a)
            # plt.title(f'clustering of {a}, {score}:{clustering_score_a}')
                print(f'{score}of {a}:{clustering_score_a}')
            plt.title(f'clustering of {a}')

            plt.subplot(2, 3, 4)
            reduced_features_b, _ = get_embs_visualization(b, True)
            plt.title(f'{b} low-level features')

            plt.subplot(2, 3, 5)
            get_seg_visualization(b)
            plt.title(f'{b} prediction mIou=0.6247')

            plt.subplot(2, 3, 6)

            clustered_indices_b = cluster.fit(reduced_features_b).labels_

            if score == 'dbcv':
                clustering_score_b = dbcv(reduced_features_b, clustered_indices_b, dist_function=euclidean)
                print(f'{score}of {b}:{clustering_score_b}')

            elif score=='silhouette':
                clustering_score_b = silhouette_score(reduced_features_b, clustered_indices_b)
                print(f'{score}of {b}:{clustering_score_b}')
            plt.scatter(reduced_features_b[:, 0], reduced_features_b[:, 1], c=clustered_indices_b)

            # plt.title(f'clustering of {b},  {score}:{clustering_score_b}')
            # clustering_score_b = silhouette_score(reduced_features_b, clustered_indices_b)
            # plt.title(f'clustering of {b}, silhouette_score:{clustering_score_b}')
            plt.title(f'clustering of {b}')
            plt.show()
        elif method == 'kmeans':
            plt.subplot(2, 3, 1)
            reduced_features_a, reduced_indices_a = get_embs_visualization(a, True)
            plt.title(f'{a} low-level features')

            plt.subplot(2, 3, 2)
            get_seg_visualization(a)
            plt.title(f'{a} prediction mIou=0.61')

            plt.subplot(2, 3, 3)
            cluster = KMeans(n_clusters=len(np.unique(reduced_indices_a)), max_iter=1000)
            # cluster = KMeans(n_clusters=5, max_iter=1000)
            clustered_indices_a = cluster.fit(reduced_features_a).labels_
            clustering_score_a = silhouette_score(reduced_features_a, clustered_indices_a)
            plt.scatter(reduced_features_a[:, 0], reduced_features_a[:, 1], c=clustered_indices_a)
            plt.title(f'clustering of {a}, silhouette score:{clustering_score_a}')

            plt.subplot(2, 3, 4)
            reduced_features_b, reduced_indices_b = get_embs_visualization(b, True)
            plt.title(f'{b} low-level features')

            plt.subplot(2, 3, 5)
            get_seg_visualization(b)
            plt.title(f'{b} prediction mIou=0.62')

            plt.subplot(2, 3, 6)
            cluster = KMeans(n_clusters=len(np.unique(reduced_indices_b)), max_iter=2000)
            # cluster = KMeans(n_clusters=5, max_iter=1000)

            clustered_indices_b = cluster.fit(reduced_features_b).labels_
            clustering_score_b = silhouette_score(reduced_features_b, clustered_indices_b)
            plt.scatter(reduced_features_b[:, 0], reduced_features_b[:, 1], c=clustered_indices_b)
            plt.title(f'clustering of {b}, silhouette score:{clustering_score_b}')

            plt.show()
        else:
            raise ValueError('please select a clustering method between dbscan and kmeans')


# compare_ll_feats('spc_focal', 'focal_spc_focal_spc', True, 'dbscan','dbcv')
get_seg_visualization('spc_focal')
plt.show()
# get_embs_visualization('spc_spc', False)
