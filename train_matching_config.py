"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

DATASET_DIR = 'C:/Users/iml/Desktop/CropMatch/DATA/Labeled'  # train(student&teacher) dir
DATASET_NAME = 'CornDataset'

#DATASET_DIR = 'F:/cityscapes/teacher'  # train(student&teacher) dir
#DATASET_NAME = 'CityscapesDataset'


args = dict(
    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='D:/CropMatch/train_distill/weight',
    save_dir_aux='D:/CropMatch/save/result3',
    save_dir1='D:/CropMatch/save/result1',
    save_dir2='D:/CropMatch/save/result2',

    distillation_path=None,

    # resume training network
    resume_path=None,
    #pretrained_path='D:/CropMatch/train_ce/erfnet3/weight/best_disease_model_387.pth',
    # pretrained_path='D:/CropMatch/train_ce/erfnet4/weight/best_disease_model_724.pth',
    pretrained_path='D:/CropMatch/train_ce/erfnet_original/weight/best_disease_model_674.pth',
    # pretrained_path=None,

    train_dataset={
        'name': DATASET_NAME,
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'train',
            'size': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'label_all', 'image_aug1', 'label_aug1', 'image_aug2', 'label_aug2'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.FloatTensor, torch.ByteTensor,
                                 torch.FloatTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 3,  # 3
        'workers': 0
    },

    val_dataset={
        'name': DATASET_NAME,
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'label_all'],
                        'type': [torch.FloatTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 0
    },

    model={
        'name': "ERFNet_Semantic3",
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },

    teacher_model={
        'name': "ERFNet_Semantic_Original",
        'kwargs': {
            'num_classes': 3,  # 3 for bg/plant/disease
        }
    },

    lr=1e-3,
    n_epochs=800,  # every x epochs, report train & validation set

)


def get_args():
    return copy.deepcopy(args)
