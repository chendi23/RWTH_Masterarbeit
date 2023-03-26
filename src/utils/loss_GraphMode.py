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
"""Loss function"""
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import stop_gradient

class CrossEntropyLoss(nn.Cell):
    """CrossEntropyLoss"""

    def __init__(self,
                 ignore_label=255):
        super(CrossEntropyLoss, self).__init__()

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

        self.not_equal = ops.NotEqual()
        self.equal = ops.Equal()

        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(False)
        self.div = ops.RealDiv()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

        self.ignore_label = ignore_label

    def construct(self, logits, labels):
        """construct"""
        num_cls = logits.shape[1]
        labels_int32 = self.cast(labels, mindspore.int32)
        labels_int = self.reshape(labels_int32, (-1,))
        logits_1 = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_1, (-1, num_cls))

        weights_1 = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights_1, mindspore.float32)

        _ce_loss = self.ce(logits_, labels_int)
        weighted_ce_loss = self.mul(weights, _ce_loss)
        ce_loss = self.div(self.sum(weighted_ce_loss), self.sum(weights))
        return ce_loss


class OhemCELoss(nn.Cell):
    """OhemCELoss"""

    def __init__(self,
                 thresh,
                 n_min,
                 ignore_label=255):
        super(OhemCELoss, self).__init__()

        self.cast = ops.Cast()
        self.scast = ops.ScalarCast()

        self._thresh = self.scast(thresh, mindspore.float32)
        self._n_min = n_min
        self._ignore_label = ignore_label

        self.topk = ops.TopK(sorted=True)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.not_equal = ops.NotEqual()
        self.equal = ops.Equal()
        self.min = ops.Minimum()
        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(False)
        self.div = ops.RealDiv()
        self.gather = ops.GatherNd()

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, logits, labels):
        """construct"""
        _, c, _, _ = logits.shape
        num_classes = c

        labels_0 = self.cast(labels, mindspore.int32)
        labels_1 = self.reshape(labels_0, (-1,))
        logits_0 = self.transpose(logits, (0, 2, 3, 1))
        logits_1 = self.reshape(logits_0, (-1, num_classes))

        keep_mask_0 = self.not_equal(labels_1, self._ignore_label)
        keep_mask_1 = self.cast(keep_mask_0, mindspore.float32)

        pix_losses = self.ce(logits_1, labels_1)
        masked_pixel_losses = self.mul(keep_mask_1, pix_losses)

        top_k_losses, _ = self.topk(masked_pixel_losses, self._n_min)
        thresh = self.min(self._thresh, top_k_losses[self._n_min - 1:self._n_min:1])

        ohem_mask = self.cast(masked_pixel_losses >= thresh, mindspore.float32)
        ohem_mask = stop_gradient(ohem_mask)
        ohem_loss = self.mul(ohem_mask, masked_pixel_losses)
        total_loss = self.sum(ohem_loss)
        num_present = self.sum(ohem_mask)
        loss = self.div(total_loss, num_present)

        return loss


class PixelContrastLoss(nn.Cell):
    def __init__(self, args):
        super(PixelContrastLoss, self).__init__()

        self.args = args
        self.temperature = self.args.contrast_temperature
        self.base_temperature = self.args.contrast_base_temperature

        self.ignore_label = self.args.ce_ignore_index

        self.max_samples = self.args.contrast_max_samples
        self.max_views = self.args.contrast_max_views

    def _hard_anchor_sampling(self, X, y_hat, y_hat_unique,y):
        # print('labels:', y_hat.shape, 'preds:', y.shape)
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        # y_hat = ops.cast(y_hat, mindspore.float32)
        # y = ops.cast(y, mindspore.float32)
        for ii in range(batch_size):
            # this_y = ops.slice(y_hat, (ii+1,0),(ii+1,-1))
            this_y = y_hat[ii]
            this_classes = [x for x in y_hat_unique[ii] if (x != self.ignore_label and x!=0)]
            # this_classes = [x for x in this_classes if (this_y ==x).nonzero().shape[0] > self.max_views]
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)
        X_ = ops.zeros((total_classes, n_view, feat_dim), mindspore.float32)
        y_ = ops.zeros(total_classes, mindspore.float32)

        X_ptr = 0

        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            # print('this class:', this_classes)


            for cls_id in this_classes:
                hard_indices = ops.logical_and(this_y_hat == cls_id, this_y != cls_id).nonzero()
                easy_indices = ops.logical_and(this_y_hat == cls_id, this_y == cls_id).nonzero()

                num_hard = hard_indices.size
                num_easy = easy_indices.size

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                # perm = ops.Randperm(max_length=num_hard, pad=-1)(ops.Tensor([num_hard], mindspore.int32))
                perm = np.random.permutation(num_hard)
                perm = ops.Tensor(perm)

                hard_indices = hard_indices[perm[:num_hard_keep]]


                if num_easy_keep != 0:
                    # perm = ops.Randperm(max_length=num_easy, pad=-1)(ops.Tensor([num_easy], mindspore.int32))
                    perm = np.random.permutation(num_easy)
                    perm = ops.Tensor(perm)

                    easy_indices = easy_indices[perm[:num_easy_keep]]


                    indices = ops.concat((hard_indices, easy_indices), 0)
                else:
                    indices = hard_indices

                X_[X_ptr] = ops.scatter_update(input_x=X_[X_ptr],updates=X[ii, indices].squeeze(1), indices=ops.arange(indices.size))
                # X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1
        
        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.view(-1, 1)
        mask = ops.equal(labels_, ops.transpose(labels_, (1, 0)))
        contrast_count = n_view
        contrast_feature = ops.concat(ops.unstack(feats_, 1), 0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = ops.div(ops.matmul(anchor_feature, ops.transpose(contrast_feature, (1, 0))),
                                      self.temperature)

        _, logits_max = ops.max(anchor_dot_contrast, axis=1, keep_dims=True)
        logits_max = stop_gradient(logits_max)
        logits = anchor_dot_contrast - logits_max

        mask = ops.tile(mask, (anchor_count, contrast_count))
        mask = stop_gradient(mask)
        mask = ops.cast(mask, mindspore.int8)

        neg_mask = 1 - mask


        ###TODO
        logits_mask = ops.eye(anchor_count * anchor_num, mask.shape[0], mindspore.bool_)
        logits_mask = stop_gradient(logits_mask)
        logits_mask = ops.cast(~logits_mask, mindspore.uint8)

        mask = mask * logits_mask


        ###
        neg_logits = ops.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1)

        exp_logits = ops.exp(logits)

        log_prob = logits - ops.log(exp_logits + neg_logits)

        # print(logits,ops.log(exp_logits + neg_logits))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # print(mean_log_prob_pos)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()


        return loss

    def construct(self, feats, labels, predict):
        # labels = ops.expand_dims(labels, 1)
        #
        # labels = ops.interpolate(labels,
        #                          sizes=(feats.shape[2], feats.shape[3]), mode='bilinear')
        # labels = labels.squeeze(axis=1)
        predict = ops.cast(ops.max(predict, axis=1)[0], mindspore.int16)
        labels = ops.cast(labels, mindspore.float16)
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.view(batch_size, -1)
        predict = predict.view(batch_size, -1)
        feats = feats.transpose(0, 2, 3, 1)
        feats = feats.view(feats.shape[0], -1, feats.shape[-1])
        labels_unique = np.unique(labels.asnumpy(),axis=1)
        labels_unique = np.pad(labels_unique, (2 - len(labels_unique)), 'constant', constant_values=0)
        labels_unique = ops.Tensor(labels_unique)
        feats_, labels_ = self._hard_anchor_sampling(feats, labels,labels_unique,predict)
        loss = self._contrastive(feats_, labels_)
        return loss


class Contrast_OHEMCE(nn.Cell):
    def __init__(self, args):
        super(Contrast_OHEMCE, self).__init__()
        self.contrast_loss = PixelContrastLoss(args)
        self.OHEMCE_loss = OhemCELoss(args.thresh, args.n_min, 255)
        self.contrast_weight = args.contrast_loss_weight

    def construct(self, emb, labels, out):
        contrast_part = self.contrast_weight * self.contrast_loss(emb, labels, out)
        ohemce_part = self.OHEMCE_loss(out, labels)
        total_loss = contrast_part + ohemce_part
        return total_loss


def build_criterion(args):
    """build_criterion"""
    print("=> Trying build {:}loss".format(args.criterion))
    if args.criterion == 'ce':
        loss = CrossEntropyLoss(ignore_label=args.ignore_label)
    elif args.criterion == 'ohemce':
        loss = OhemCELoss(args.thresh, args.n_min, args.ignore_label)
    elif args.criterion == 'contrastive_ohemce':
        loss = Contrast_OHEMCE(args)
    else:
        raise ValueError('unknown criterion : {:}'.format(args.criterion))
    return loss


def test():
    import torch
    import torch.nn as nn_torch
    from abc import ABC

    class PixelContrastLoss_torch(nn_torch.Module, ABC):
        def __init__(self, args):
            super(PixelContrastLoss_torch, self).__init__()

            self.args = args
            self.temperature = self.args.contrast_temperature
            self.base_temperature = self.args.contrast_base_temperature

            self.ignore_label = self.args.ce_ignore_index

            self.max_samples = self.args.contrast_max_samples
            self.max_views = self.args.contrast_max_views

        def _hard_anchor_sampling(self, X, y_hat, y):
            batch_size, feat_dim = X.shape[0], X.shape[-1]

            classes = []
            total_classes = 0
            for ii in range(batch_size):
                this_y = y_hat[ii]
                this_classes = torch.unique(this_y)
                this_classes = [x for x in this_classes if x != self.ignore_label]
                this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

                classes.append(this_classes)
                total_classes += len(this_classes)

            if total_classes == 0:
                return None, None

            n_view = self.max_samples // total_classes
            n_view = min(n_view, self.max_views)
            X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float)
            y_ = torch.zeros(total_classes, dtype=torch.float)

            X_ptr = 0
            for ii in range(batch_size):
                this_y_hat = y_hat[ii]
                this_y = y[ii]
                this_classes = classes[ii]

                for cls_id in this_classes:
                    hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                    easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                    num_hard = hard_indices.shape[0]
                    num_easy = easy_indices.shape[0]
                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        # Log.log('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                        raise Exception
                    np.random.seed(1)
                    perm = np.random.permutation(num_hard)
                    # perm = torch.randperm(num_hard)
                    # print(hard_indices, perm, num_hard_keep)

                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    # print(hard_indices)

                    perm = np.random.permutation(num_easy)
                    # perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)
                    X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                    y_[X_ptr] = cls_id
                    X_ptr += 1

            return X_, y_

        def _contrastive(self, feats_, labels_):
            anchor_num, n_view = feats_.shape[0], feats_.shape[1]

            labels_ = labels_.contiguous().view(-1, 1)
            mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float()

            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

            anchor_feature = contrast_feature
            anchor_count = contrast_count

            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                            self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            # print('anchor_dot:', anchor_dot_contrast, 'logit_max:', logits_max)
            logits = anchor_dot_contrast - logits_max.detach()

            mask = mask.repeat(anchor_count, contrast_count)
            neg_mask = 1 - mask

            logits_mask = torch.ones_like(mask).scatter_(1,
                                                         torch.arange(anchor_num * anchor_count).view(-1, 1),
                                                         0)
            mask = mask * logits_mask
            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)

            log_prob = logits - torch.log(exp_logits + neg_logits)
            # print(logits, torch.log(exp_logits + neg_logits))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            # print(mean_log_prob_pos)

            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

            return loss

        def forward(self, feats, labels=None, predict=None):
            predict = torch.argmax(predict, 1)
            labels = labels.unsqueeze(1).float().clone()
            labels = torch.nn.functional.interpolate(labels,
                                                     (feats.shape[2], feats.shape[3]), mode='nearest')
            labels = labels.squeeze(1).long()
            assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

            batch_size = feats.shape[0]

            labels = labels.contiguous().view(batch_size, -1)
            predict = predict.contiguous().view(batch_size, -1)
            feats = feats.permute(0, 2, 3, 1)
            feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
            loss = self._contrastive(feats_, labels_)

            return loss

    np.random.seed(1)
    from src.config import obtain_autodeeplab_args
    from mindspore import context
    args = obtain_autodeeplab_args()
    args.contrast_max_views = 10
    args.contrast_max_samples = 100
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target='CPU',
                        device_id=0)
    labels_np = np.random.randint(0,2, (2, 32, 32))
    labels = ops.Tensor(labels_np, mindspore.float32)
    labels_torch = torch.from_numpy(labels_np)
    feats_np = np.random.normal(0, 1, size=[2, 16, 32, 32])
    feats = ops.Tensor(feats_np, mindspore.float32)
    feats_torch = torch.Tensor(feats_np)
    predict_np = np.random.randint(0, 4, [2, 4, 32, 32])
    predict = ops.Tensor(predict_np, mindspore.float32)
    predict_torch = torch.Tensor(predict_np)
    contrast_loss = PixelContrastLoss(args)
    contrast_loss_torch = PixelContrastLoss_torch(args)
    print(contrast_loss(feats, labels, predict))
    print(contrast_loss_torch(feats_torch, labels_torch, predict_torch))


if __name__ == "__main__":
    test()
