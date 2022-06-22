import argparse

from FS2K_dataset import FS2KData as D
import utils_image as util
from PIL import Image
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="PyTorch FS2K")
parser.add_argument("--attribute", type=str, default='gender', help="attribute of face")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")

global opt
opt = parser.parse_args()


# seed = random.randint(1, 10000)
# print('Random seed: {}'.format(seed))
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

file_path_train = 'FS2K/train/photo'
file_path_test = 'FS2K/test/photo'
json_file_train = 'FS2K/anno_train.json'
json_file_test = 'FS2K/anno_test.json'

# 定义数据集
train_set = D(json_file_train, file_path_train, 'train', attribute=opt.attribute)
train_loader = DataLoader(train_set,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)


test_set = D(json_file_test, file_path_test, 'test', attribute=opt.attribute)
test_loader = DataLoader(test_set,
                         batch_size=16,
                         shuffle=False,
                         num_workers=0,
                         drop_last=False,
                         pin_memory=True)



# for i, train_data in enumerate(train_loader):
#     print(train_data[0].shape, train_data[1].shape)
#     break

# 定义网络
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def res_model(num_classes, feature_extract=False, use_pretrained=False):
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


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
learning_rate = 1e-4
weight_decay = 1e-3
num_epoch = 50
model_path = 'pre_res_model_' + opt.attribute + '.ckpt'

model = res_model(opt.num_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
n_epochs = num_epoch
best_acc = 0.0


writer = SummaryWriter("res34_" + opt.attribute)
for epoch in range(n_epochs):
    model.train()
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        # print(logits.shape, labels.shape)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_acc", train_acc, epoch)


    model.eval()
    test_loss = []
    test_accs = []
    for batch in tqdm(test_loader):
        imgs, labels = batch
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        test_loss.append(loss.item())
        test_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_accs) / len(test_accs)
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_acc", test_acc, epoch)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))