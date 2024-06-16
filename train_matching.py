"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import train_distill_config

from criterions.loss_matching import CriterionCE, CriterionDistillationMatching
from datasets import get_dataset
from models import get_model, ERFNet_Semantic_Original
from utils.utils import AverageMeter, Logger, Visualizer  # for CVPPP


torch.backends.cudnn.benchmark = True

args = train_matching_config.get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    if not os.path.exists(args['save_dir1']):
        os.makedirs(args['save_dir1'])
    if not os.path.exists(args['save_dir2']):
        os.makedirs(args['save_dir2'])
    if not os.path.exists(args['save_dir_aux']):
        os.makedirs(args['save_dir_aux'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# train dataloader (student)
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader (student)
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=False, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set criterion
criterion_val = CriterionCE()
criterion = CriterionDistillationMatching(num_classes=3)

criterion_val = torch.nn.DataParallel(criterion_val).to(device)
criterion = torch.nn.DataParallel(criterion).to(device)

# Logger
logger = Logger(('train', 'train_ce_loss', 'train_matching_loss',
                 'val', 'val_iou_plant', 'val_iou_disease'), 'loss')


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


def save_checkpoint(epoch, state, recon_best1, recon_best2, recon_best3, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if recon_best1:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_plant_model_%d.pth' % (epoch)))

    if recon_best2:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_disease_model_%d.pth' % (epoch)))

    if recon_best3:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_both_model_%d.pth' % (epoch)))


# Transform functions
import random
import torchvision
#import kornia

# ---------------------------------------- Augmentation Methods ----------------------------------------
def random_scale_crop(scale, data=None, target=None, ignore_label=255, probs=None):
    """

    Args:
        scale: scale ratio. Float
        data:  input data to augment BxCxWxH
        target: labels to augment BxWxH
        probs: probability masks to augment BxCxWxH
        ignore_label: integeer value that defines the ignore class in the datasets for the labels

    Returns:
         data, target and prob, after applied a scaling operation. output resolution is preserve as the same as the input resolution  WxH
    """
    if scale != 1:
        init_size_w = data.shape[2]
        init_size_h = data.shape[3]

        # scale data, labels and probs
        data = nn.functional.interpolate(data, scale_factor=scale, mode='bilinear', align_corners=True,
                                         recompute_scale_factor=True)
        if target is not None:
            target = nn.functional.interpolate(target.unsqueeze(1).float(), scale_factor=scale, mode='nearest',
                                               recompute_scale_factor=True).long().squeeze(1)
        if probs is not None:
            probs = nn.functional.interpolate(probs, scale_factor=scale, mode='bilinear', align_corners=True,
                                              recompute_scale_factor=True).squeeze(1)

        final_size_w = data.shape[2]
        final_size_h = data.shape[3]
        diff_h = init_size_h - final_size_h
        diff_w = init_size_w - final_size_w
        if scale < 1:  # add padding if needed
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w // 2, diff_w // 2 + 1, diff_h // 2 + 1, diff_h // 2), 0)
            else:
                pad = nn.ConstantPad2d((diff_w // 2, diff_w // 2, diff_h // 2, diff_h // 2), 0)

            data = pad(data)
            if probs is not None:
                probs = pad(probs)

            # padding with ignore label to add to labels
            if diff_h % 2 == 1:
                pad = nn.ConstantPad2d((diff_w // 2, diff_w // 2 + 1, diff_h // 2 + 1, diff_h // 2), ignore_label)
            else:
                pad = nn.ConstantPad2d((diff_w // 2, diff_w // 2, diff_h // 2, diff_h // 2), ignore_label)

            if target is not None:
                target = pad(target)

        else:  # crop if needed
            w = random.randint(0, data.shape[2] - init_size_w)
            h = random.randint(0, data.shape[3] - init_size_h)
            data = data[:, :, h:h + init_size_h, w:w + init_size_w]
            if probs is not None:
                probs = probs[:, h:h + init_size_h, w:w + init_size_w]
            if target is not None:
                target = target[:, h:h + init_size_h, w:w + init_size_w]

    return data, target, probs


def transform(image, label, embeddings):
    '''image: (N, 3, H, W) / label: (N, H, W) / embeddings: (N, C, H, W) = model(image)
       image_aug=T(image) / label_aug=T(label) / embeddings_trans=T(embeddings)

       Objective: embeddings_trans = model(image_aug)'''

    image_aug = image.clone()
    label_aug = label.clone()

    embeddings_trans = embeddings.clone()

    transform_list = []

    if random.random() > 0.5:
        #transform_list.append('hflip')

        image_aug = torch.flip(image_aug, [3])
        label_aug = torch.flip(label_aug, [2])

        embeddings_trans = torch.flip(embeddings_trans, [3])

    if random.random() > 0.5:
        #transform_list.append('vflip')

        image_aug = torch.flip(image_aug, [2])
        label_aug = torch.flip(label_aug, [1])

        embeddings_trans = torch.flip(embeddings_trans, [2])

    if random.random() > 0.5:
        #transform_list.append('rotate90')

        image_aug = torchvision.transforms.functional.rotate(image_aug, angle=90)
        label_aug = torchvision.transforms.functional.rotate(label_aug, angle=90)

        embeddings_trans = torchvision.transforms.functional.rotate(embeddings_trans, angle=90)

    if random.random() > 0.5:
        #transform_list.append('rotate180')

        image_aug = torchvision.transforms.functional.rotate(image_aug, angle=180)
        label_aug = torchvision.transforms.functional.rotate(label_aug, angle=180)

        embeddings_trans = torchvision.transforms.functional.rotate(embeddings_trans, angle=180)

    if random.random() > 0.5:
        #transform_list.append('rotate270')

        image_aug = torchvision.transforms.functional.rotate(image_aug, angle=270)
        label_aug = torchvision.transforms.functional.rotate(label_aug, angle=270)

        embeddings_trans = torchvision.transforms.functional.rotate(embeddings_trans, angle=270)

    if random.random() > 0.5:
        transform_list.append('randomscalecrop')

        scale = random.uniform(0.8, 1.0)

        image_aug, label_aug, embeddings_trans = random_scale_crop(scale=scale,
                                                                   data=image_aug, target=label_aug,
                                                                   probs=embeddings_trans)
    # if random.random() > 0.8:
    #     transform_list.append('blurring')
    #     blur = nn.Sequential(kornia.filters.GaussianBlur2d(kernel_size=(23, 23), sigma=(0.2, 3)))
    #
    #     image_aug = blur(image_aug)

    return image_aug, label_aug, embeddings_trans


def main():
    # init
    start_epoch = 0
    best_iou_plant = 0
    best_iou_disease = 0
    best_iou_both = 0

    # set model (student)
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)
    if args['pretrained_path']:
        state = torch.load(args['pretrained_path'])
        model.load_state_dict(state['model_state_dict'], strict=False)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=1e-4)

    def lambda_(epoch):
        return pow((1 - ((epoch) / args['n_epochs'])), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_, )

    # resume (student)
    if args['resume_path'] is not None and os.path.exists(args['resume_path']):
        print('Resuming model-student from {}'.format(args['resume_path']))
        state = torch.load(args['resume_path'])
        start_epoch = state['epoch'] + 1
        best_iou_plant = state['best_iou_plant']
        best_iou_disease = state['best_iou_disease']
        best_iou_both = state['best_iou_both']
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']

    for epoch in range(start_epoch, args['n_epochs']):
        print('Starting epoch {}'.format(epoch))

        loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_matching_meter = AverageMeter()

        # Training (Student)
        for i, sample in enumerate(tqdm(train_dataset_it)):
            image = sample['image']  # (N, 3, 512, 512)
            label = sample['label_all'].squeeze(1)  # (N, 512, 512)

            # ------------------------ forward ------------------------
            model.train()
            outputs, embeddings = model(image)  # (N, num_classes=3, 512, 512), # (N, 32, 512, 512)

            # ------------------------ make augmented sets ------------------------
            # image_aug=T(image) / label_aug=T(label) / embeddings_trans=T(model(image)), T=transform
            image_aug, label_aug, embeddings_trans = transform(image, label, embeddings)
            image_aug = torchvision.transforms.functional.adjust_sharpness(image_aug, 10)

            # outputs_aug, embeddings_aug = model(T(image))
            outputs_aug, embeddings_aug = model(image_aug)

            # ------------------------ calculate loss ------------------------
            # objective: model(T(image)) = T(model(image)) at feature level
            loss, loss_ce, loss_matching = \
                criterion(outputs=outputs, class_labels=label, outputs_aug=outputs_aug, class_labels_aug=label_aug,
                          embeddings_trans=embeddings_trans, embeddings_aug=embeddings_aug)

            loss = loss.mean()
            loss_ce = loss_ce.mean()
            loss_matching = loss_matching.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            loss_ce_meter.update(loss_ce.item())
            loss_matching_meter.update(loss_matching.item())

        train_loss, train_ce, train_matching= \
            loss_meter.avg, loss_ce_meter.avg, loss_matching_meter.avg
        scheduler.step()

        print('===> train loss: {:.5f}, train-ce: {:.5f}, train-matching: {:.5f}' \
              .format(train_loss, train_ce, train_matching))
        logger.add('train', train_loss)
        logger.add('train_ce_loss', train_ce)
        logger.add('train_matching_loss', train_matching)

        # validation
        loss_val_meter = AverageMeter()
        iou1_meter, iou2_meter = AverageMeter(), AverageMeter()

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(val_dataset_it)):
                image = sample['image']  # (N, 3, 512, 512)
                label = sample['label_all'].squeeze(1)  # (N, 512, 512)

                output, _ = model(image)  # (N, 4, h, w)

                loss = criterion_val(output, label,
                                     iou=True, meter_plant=iou1_meter, meter_disease=iou2_meter)
                loss = loss.mean()
                loss_val_meter.update(loss.item())

        val_loss, val_iou_plant, val_iou_disease = loss_val_meter.avg, iou1_meter.avg, iou2_meter.avg
        print('===> val loss: {:.5f}, val iou-plant: {:.5f}, val iou-disease: {:.5f}'.format(val_loss, val_iou_plant,
                                                                                             val_iou_disease))

        logger.add('val', val_loss)
        logger.add('val_iou_plant', val_iou_plant)
        logger.add('val_iou_disease', val_iou_disease)
        logger.plot(save=args['save'], save_dir=args['save_dir'])

        # save
        is_best_plant = val_iou_plant > best_iou_plant
        best_iou_plant = max(val_iou_plant, best_iou_plant)

        is_best_disease = val_iou_disease > best_iou_disease
        best_iou_disease = max(val_iou_disease, best_iou_disease)

        val_iou_both = (val_iou_plant + val_iou_disease) / 2

        is_best_both = val_iou_both > best_iou_both
        best_iou_both = max(val_iou_both, best_iou_both)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou_plant': best_iou_plant,
                'best_iou_disease': best_iou_disease,
                'best_iou_both': best_iou_both,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data
            }
            save_checkpoint(epoch, state, is_best_plant, is_best_disease, is_best_both)


if __name__ == '__main__':
    main()






