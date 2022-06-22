import cv2
import torch.utils.data as data
import utils_image as util
import random
import json

class FS2KData(data.Dataset):
    def __init__(self, json_path, img_path, mode='train', resize_height=128, resize_width=128, attribute='gender'):
        self.mode = mode
        self.paths = util.get_image_paths(img_path)
        with open(json_path, 'r') as f:
            self.json_data = json.loads(f.read())
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.attribute = attribute

    def __getitem__(self, index):
        path = self.paths[index]
        img = util.imread_uint(path)
        img = cv2.resize(img, (self.resize_height, self.resize_width))
        img = util.uint2single(img)

        # if self.mode == 'train':
        #     # 给训练集加上旋转/翻转的数据增强
        #     m = random.randint(0, 7)
        #     img = util.augment_img(img, mode=m)

        # 图片
        img = util.single2tensor3(img)
        if self.mode == 'train':
            # 对应label
            old_img_name = self.paths[index][17:]
            new_img_name = old_img_name[:6] + '/' + old_img_name[7:-4]
            i = 0
            while i < len(self.paths):
                if self.json_data[i]['image_name'] == new_img_name:
                    # hair_color = self.json_data[i]['hair_color']
                    # hair= self.json_data[i]['hair']
                    # gender = self.json_data[i]['gender']
                    # earring = self.json_data[i]['earring']
                    label = self.json_data[i][self.attribute]
                    break
                i = i + 1

        if self.mode == 'test':
            # 对应label
            old_img_name = self.paths[index][16:]
            new_img_name = old_img_name[:6] + '/' + old_img_name[7:-4]
            i = 0
            while i < len(self.paths):
                if self.json_data[i]['image_name'] == new_img_name:
                    # hair_color = self.json_data[i]['hair_color']
                    # hair = self.json_data[i]['hair']
                    # gender = self.json_data[i]['gender']
                    # earring = self.json_data[i]['earring']
                    label = self.json_data[i][self.attribute]
                    break
                i = i + 1

        return img, label

    def __len__(self):
        return len(self.paths)