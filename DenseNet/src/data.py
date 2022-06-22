import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
import json
from collections.abc import Iterable


def _get_feature_list(feature_list, raw_slice):
    res = []
    for feature in feature_list:
        if isinstance(raw_slice[feature], Iterable):
            for idx, value in enumerate(raw_slice[feature]):
                res.append(value)
        else:
            res.append(raw_slice[feature])
    return res


class FS2KData(Dataset):
    def __init__(self,
                 json_path,
                 img_path,
                 mode='train',
                 resize_height=256,
                 resize_width=256,
                 feature_list=None):

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        if feature_list is None:
            feature_list = ['gender']
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.img_path = img_path
        self.json_path = json_path
        self.mode = mode

        # 读取 json 文件
        f = open(self.json_path, 'r', encoding='utf-8')
        self.json_data = json.load(f)

        if mode == 'train':
            # 获取json中图像的名称 属性
            self.train_image = []
            self.train_label = []

            for i in range(len(self.json_data)):
                self.train_image.append(self.json_data[i]['image_name'])
                arr = np.array(_get_feature_list(feature_list, self.json_data[i]))
                self.train_label.append(arr)


            self.image_arr = self.train_image
            self.label_arr = self.train_label


        elif mode == 'test':
            self.test_image = []
            self.test_label = []

            for i in range(len(self.json_data)):
                self.test_image.append(self.json_data[i]['image_name'])
                arr = np.array(_get_feature_list(feature_list, self.json_data[i]))
                self.test_label.append(arr)
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)




    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.img_path + '/' + single_image_name.replace('/image', '_image') + '.png')

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            label = self.test_label[index]

            return img_as_img, label
        else:
            label = self.train_label[index]


            return img_as_img, label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len

