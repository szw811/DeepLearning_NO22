# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

def build_dataset(is_train, args):

    if args.data_set != "fs2k":
        transform = build_transform(is_train, args)

        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "fs2k":
        root = args.data_path if is_train else args.eval_data_path
        dataset = FS2KData(args.data_path, 'train' if is_train else 'test', args.input_size, args.input_size)
        nb_classes = args.nb_classes
        # assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
            transforms.Resize((args.input_size, args.input_size),
                            interpolation=transforms.InterpolationMode.BICUBIC),
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class FS2KData(Dataset):
    def __init__(self, data_path, mode='train', resize_height=256, resize_width=256, imagenet_default_mean_and_std=None, labels=None):
        if labels is None or len(labels) == 0:
            self.labels = ["hair"]
        else:
            self.labels = labels
        # ??????????????????????????????????????????????????????????????????????????????#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.img_path = os.path.join(data_path, "photo")
        # self.json_path = json_path
        self.mode = mode
        self.mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        if mode == 'train':
            # ??????json?????????????????? ??????
            self.json_path = os.path.join(data_path, "anno_train.json")
            # ?????? json ??????
        elif mode == 'test':
            # ??????json?????????????????? ??????
            self.json_path = os.path.join(data_path, "anno_test.json")
        self.json_data = self.load_json_data()
        self.image_arr = []
        self.label_arr = []

        for i in range(len(self.json_data)):
            self.image_arr.append(self.json_data[i]['image_name'])

            self.label_arr.append(self.json_data[i]['earring'])

        self.real_len = len(self.image_arr)

    def load_json_data(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        return json_data

    def __getitem__(self, index):
        # ??? image_arr?????????????????????????????????
        single_image_name = self.image_arr[index]

        # ??????????????????
        img_as_img = Image.open(os.path.join(self.img_path, single_image_name + '.jpg'))

        # ???????????????RGB????????????????????????????????????????????????????????????
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # ????????????????????????????????????????????????????????????nomarlize????????????
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                # transforms.RandomHorizontalFlip(p=0.5),  # ?????????????????? ??????????????????
                transforms.ToTensor(),
                # transforms.Normalize(self.mean, self.std),
            ])
        else:
            # valid???test??????????????????
            transform = transforms.Compose([
                transforms.Resize((self.resize_height, self.resize_width)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)
        label = self.label_arr[index]

        return img_as_img, label  # ???????????????index?????????????????????????????????label

    def __len__(self):
        return self.real_len
