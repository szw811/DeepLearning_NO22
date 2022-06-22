import argparse

import matplotlib
# % matplotlib inline
import torchvision.models as models
from FS2K_dataset import FS2KData as D
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from model import Mod_Resnet
from data import FS2KData

parser = argparse.ArgumentParser(description="PyTorch FS2K")
parser.add_argument("--attribute", type=str, default='earring', help="attribute of face")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")

global opt
opt = parser.parse_args()

# 类
if opt.attribute == 'gender':
    classes = ['male', 'female']
elif opt.attribute == 'hair_color':
    classes = ['brown', 'black', 'red', 'no-hair', 'golden']
elif opt.attribute == 'hair':
    classes = ['with hair', 'without hair']
elif opt.attribute == 'earring':
    classes = ['with earring', 'without earring']

# 混淆矩阵函数
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

# 定义网络
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    # 设置最后全连接层的输出
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, num_classes),
                                nn.ReLU(),
                                )
    return model_ft


# model = res_model(opt.num_classes)
model = Mod_Resnet()
model_path = './pre_res_model_Mod_Resnet.ckpt'
# model_path = 'pre_res_model_' + opt.attribute + '.ckpt'
# create model and load weights from checkpoint
model = model.to(device)
model.load_state_dict(torch.load(model_path))

model.eval()


# file_path_test = 'FS2K/test/photo'
# json_file_test = 'FS2K/anno_test.json'

file_path_test = '../FS2K/test/photo'
json_file_test = '../FS2K/anno_test.json'

# 定义测试数据集
# test_set = D(json_file_test, file_path_test, 'test', attribute=opt.attribute)
# test_loader = DataLoader(test_set,
#                          batch_size=16,
#                          shuffle=False,
#                          num_workers=0,
#                          drop_last=False,
#                          pin_memory=True, )
batch_size = 32
test_set = FS2KData(json_file_test, file_path_test, 'test')
test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True, )

# 测试
predictions = []
labels = []
for idx, batch in enumerate(tqdm(test_loader)):
    imgs, label = batch
    with torch.no_grad():
        _,_,_,logits = model(imgs.to(device))
    # 记录原标签和预测值
    if idx == 0:
        print(logits, label[0])
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    labels.extend(label[:,3].cpu().numpy())
    if idx == 0:
        print(predictions, labels)

# 生成混淆矩阵
cm = confusion_matrix(labels, predictions)
plot_confusion_matrix(cm, 'confusion_matrix_' + opt.attribute + '.png', title='confusion matrix')


# print(test_set.paths[0])
# old_img_name = test_set.paths[0][16:]
# new_img_name = old_img_name[:6] + '/' + old_img_name[7:-4]
# print(new_img_name)
# i = 0
# while i < len(test_set.paths):
#     if test_set.json_data[i]['image_name'] == new_img_name:
#         hair_color = test_set.json_data[i]['hair_color']
#         break
#     i = i + 1
# print(hair_color)
