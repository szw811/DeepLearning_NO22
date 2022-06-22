import torch

file_path_train = '../FS2K/train/photo'
file_path_test = '../FS2K/test/photo'
json_file_train = '../FS2K/anno_train.json'
json_file_test = '../FS2K/anno_test.json'
model_path = '../model/pre_res_model_ModelDenseNet.ckpt'

feature_dict = {
    'skin_color': {
        'iterable': True,
        'value': [
            {
                'key': 'skin_color_0',
                'type': torch.int32,
                'linear_out': 255
            },
            {
                'key': 'skin_color_1',
                'type': torch.int32,
                'linear_out': 255
            }
        ],
        'loss_weight': 0.2
    },
    'lip_color': {
        'iterable': True,
        'value': [
            {
                'key': 'lip_color_0',
                'type': torch.float32,
                'linear_out': -1
            },
            {
                'key': 'lip_color_1',
                'type': torch.float32,
                'linear_out': -1
            },
            {
                'key': 'lip_color_2',
                'type': torch.float32,
                'linear_out': -1
            },
        ],
        'loss_weight': 0.2
    },
    'eye_color': {
        'iterable': True,
        'value': [
            {
                'key': 'eye_color_0',
                'type': torch.float32,
                'linear_out': -1
            },
            {
                'key': 'eye_color_1',
                'type': torch.float32,
                'linear_out': -1
            },
            {
                'key': 'eye_color_2',
                'type': torch.float32,
                'linear_out': -1
            }
        ],
        'loss_weight': 0.2
    },
    'hair': {
        'iterable': False,
        'key': 'hair',
        'type': torch.int8,
        'linear_out': 2,
        'loss_weight': 0.3
    },
    'hair_color': {
        'iterable': False,
        'key': 'hair_color',
        'type': torch.int8,
        'linear_out': 5,
        'loss_weight': 1
    },
    'gender': {
        'iterable': False,
        'key': 'gender',
        'type': torch.int8,
        'linear_out': 2,
        'loss_weight': 0.5
    },
    'earring': {
        'iterable': False,
        'key': 'earring',
        'type': torch.int8,
        'linear_out': 2,
        'loss_weight': 0.5
    },
    'smile': {
        'iterable': False,
        'key': 'smile',
        'type': torch.int8,
        'linear_out': 2,
        'loss_weight': 1
    },
    'frontal_face': {
        'iterable': False,
        'key': 'frontal_face',
        'type': torch.int8,
        'linear_out': 2,
        'loss_weight': 0.3
    },
    'style': {
        'iterable': False,
        'key': 'style',
        'type': torch.int8,
        'linear_out': 3,
        'loss_weight': 1
    }
}
feature_list = [
    {
        'key': 'skin_color',
    },
    {
        'key': 'lip_color',
    },
    {
        'key': 'eye_color',
    },
    {
        'key': 'hair',
    },
    {
        'key': 'hair_color',
    },
    {
        'key': 'gender',
    },
    {
        'key': 'earring',
    },
    {
        'key': 'smile',
    },
    {
        'key': 'frontal_face',
    },
    {
        'key': 'style',
    }
]
feature_label_list = [
    {
        'key': 'skin_color_0',
        'parent': 'skin_color'
    },
    {
        'key': 'skin_color_1',
        'parent': 'skin_color'
    },
    {
        'key': 'lip_color_0',
        'parent': 'lip_color'
    },
    {
        'key': 'lip_color_1',
        'parent': 'lip_color'
    },
    {
        'key': 'lip_color_2',
        'parent': 'lip_color'
    },
    {
        'key': 'eye_color_0',
        'parent': 'eye_color'
    },
    {
        'key': 'eye_color_1',
        'parent': 'eye_color'
    },
    {
        'key': 'eye_color_2',
        'parent': 'eye_color'
    },
    {
        'key': 'hair',
        'parent': 'hair'
    },
    {
        'key': 'hair_color',
        'parent': 'hair_color'
    },
    {
        'key': 'gender',
        'parent': 'gender'
    },
    {
        'key': 'earring',
        'parent': 'earring'
    },
    {
        'key': 'smile',
        'parent': 'smile'
    },
    {
        'key': 'frontal_face',
        'parent': 'frontal_face'
    },
    {
        'key': 'style',
        'parent': 'style'
    }
]
# earring_smile_frontal_face_style.py
feature_train = [
    'earring',
    'smile',
    'frontal_face',
    'style'
]
batch_size = 128

epoch = 400

# FocalLoss 基准向量alpha
focal_loss_alpha_dict = {
    3: [0.75, 0.75, 0.75],
    5: [0.1825, 0.15, 0.235, 0.2375, 0.195]
}
# 学习率衰减方式 有[off, Exponential, CosineAnnealing, Lambda, Step]
lr_mode = 'CosineAnnealing'
# 基准学习率
lr_base = 1e-3
# Exponential 负指数学习率衰减
lr_gamma = 0.99
# CosineAnnealing 余弦退火学习率衰减
lr_t_max = 10
lr_eta_min = 0
lr_last_epoch = -1
