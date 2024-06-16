"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity


criterion_ce = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()


class CriterionCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, class_label, iou=False, meter_plant=None, meter_disease=None):
        '''embedding : embedding network output (N, 32, 512, 512)
           prediction : seg model output (N, 3, 512, 512) *3 for bg/plant/disease
           instances_all : GT plant-mask (N, 512, 512)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        loss = 0

        for b in range(0, batch_size):

            # 3.cross entropy loss
            pred = prediction[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_label[b].unsqueeze(0)  # (1, h, w)
            gt_label = gt_label.type(torch.long).cuda()

            ce_loss = criterion_ce(pred, gt_label)

            # total loss
            loss = loss + ce_loss

            if iou:
                pred = pred.detach().max(dim=1)[1]

                pred_plant = (pred == 1)
                pred_disease = (pred == 2)

                gt_plant = (class_label[b].unsqueeze(0) == 1)
                gt_disease = (class_label[b].unsqueeze(0) == 2)

                meter_plant.update(calculate_iou(pred_plant, gt_plant.cuda()))
                meter_disease.update(calculate_iou(pred_disease, gt_disease.cuda()))

        return loss


class CriterionDistillation(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, class_labels, outputs_aug, class_labels_aug,  # for cross-entropy
                embeddings_trans, embeddings_aug):  # for feature matching distillation

        '''outputs, embeddings = model(image) where outputs:(N, #classes, 512, 512) & embeddings: (N, C, 512, 512)
           outputs_aug, embeddings_aug = model(T(image)) & _, embeddings_trans = T(model(image))
           embeddings_trans should be equal to embeddings aug for consistency regularization'''

        batch_size, height, width = outputs.size(
            0), outputs.size(2), outputs.size(3)

        # ----------------- feature distillation matching loss -----------------
        loss_distill_total = criterion_mse(embeddings_trans, embeddings_aug)
        loss_distill_total = loss_distill_total / batch_size

        # ----------------- cross entropy loss -----------------
        loss_ce_total = 0

        for b in range(0, batch_size):

            # 1. for model(image)
            pred = outputs[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_labels[b].unsqueeze(0)  # (1, 512, 512)
            gt_label = gt_label.type(torch.long).cuda()

            # print(type(pred)) - torch.tensor
            # print(pred.dtype) - torch.float32
            # print(type(gt_label)) - torch.tensor
            # print(gt_label.dtype) - torch.int64
            loss_ce_total += criterion_ce(pred, gt_label)

            # 2. for model(T(image))
            pred = outputs_aug[b, :, :, :]
            pred = pred.unsqueeze(0)
            pred = pred.type(torch.float32).cuda()

            gt_label = class_labels_aug[b].unsqueeze(0)  # (1, 512, 512)
            gt_label = gt_label.type(torch.long).cuda()

            loss_ce_total += criterion_ce(pred, gt_label)

        loss_ce_total = loss_ce_total / (2*batch_size)

        loss = (1 * loss_ce_total) + (2 * loss_distill_total)
        #print('CE :', loss_ce_total)
        #print('CE :', loss_ce_total.dtype)
        #print('Matching :', loss_matching_total)

        return loss, loss_ce_total, loss_distill_total


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
