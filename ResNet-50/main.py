import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from data import FS2KData

from resnet10 import ResNet18
import torchvision.models as models
from model import Mod_Resnet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

def res_model(num_classes, feature_extract=False, use_pretrained=False):
    model_ft = models.resnet18(pretrained=use_pretrained)
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

class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
      Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
            focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.1825, 0.15, 0.235, 0.2375, 0.195], gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        # print("idx:", idx)
        # print("alpha:", alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def main():
    batch_size = 32

    # cifar_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()
    # ]), download=True)
    # print(cifar_train[0])
    # cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    #
    # cifar_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor()
    # ]), download=True)
    # print(cifar_test[0])
    # cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    file_path_train = '../FS2K/train/photo'
    file_path_test = '../FS2K/test/photo'
    json_file_train = '../FS2K/anno_train.json'
    json_file_test = '../FS2K/anno_test.json'

    train_set = FS2KData(json_file_train, file_path_train, 'train')

    cifar_train = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True, )

    test_set = FS2KData(json_file_test, file_path_test, 'test')
    cifar_test = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True, )

    x, label = iter(cifar_train).next()# iter得到DataLoader的迭代器 再用迭代器的next方法得到一个batch

    print("x: ", x.shape, "label: ", label.shape)


    device = torch.device('cuda')
    # model = ResNet18()
    # model = res_model(2)
    model = Mod_Resnet()
    model_path = './pre_res_model_Mod_Resnet.ckpt'
    if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        model = torch.nn.DataParallel(model)  # 多gpu训练,自动选择gpu
    #
    model.to(device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    criteon = nn.CrossEntropyLoss().to(device)
    criteon2 = FocalLoss(5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)# 定义一个优化器
    print(model)# 打印出网络结构

    best_acc = 0.0
    best_acc1 = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    best_acc4 = 0.0
    for epoch in range(50):
        model.train()
        for batchidx, (x, label) in enumerate(tqdm(cifar_train)):# enumerate可以生成类似元组 既得到元素又得到索引
            # x: [b, 3, 32, 32]
            # label: [b]
            x, label = x.to(device), label.to(device)

            logits_hair, logits_hair_color, logits_gender, logits_earring = model(x)# 调用Lenet5的forward方法
            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar 长度为0的标量
            loss_hair = criteon(logits_hair, label[:, 0])# 调用了criteon的forward方法
            loss_hair_color = criteon2(logits_hair_color, label[:, 1])
            loss_gender = criteon(logits_gender, label[:, 2])
            loss_earring = criteon(logits_earring, label[:, 3])
            loss = loss_hair_color + 0.25 * loss_hair + 0.3 * loss_gender + 0.2 * loss_earring

            # backprop
            optimizer.zero_grad()# 梯度清零
            # 后向的时候是做梯度累加 所以需要先清零
            loss.backward()
            optimizer.step()

        #
        # print("epoch:", epoch, " loss:", loss.item())# item将标量转化为tensor
        # print("epoch:", epoch, " loss_hair:", loss_hair.item(), " loss_hair_color:", loss_hair_color.item())
        print("epoch:", epoch, " loss_hair:", loss_hair.item(), " loss_hair_color:", loss_hair_color.item(), " loss_gender:", loss_gender.item(), " loss_earring:", loss_earring.item())

        model.eval()
        with torch.no_grad():# 告诉pytorch不需要后向传播
            #test
            # total_correct = 0
            total_correct_hair = 0
            total_correct_hair_color = 0
            total_correct_gender = 0
            total_correct_earring = 0
            total_num = 0
            for x, label in tqdm(cifar_test):

                x, label = x.to(device), label.to(device)

                # logits = model(x)
                logits_hair, logits_hair_color, logits_gender, logits_earring = model(x)
                # [b]
                # pred = logits.argmax(dim=1)
                pred_hair = logits_hair.argmax(dim=1)
                pred_hair_color = logits_hair_color.argmax(dim=1)
                pred_gender = logits_gender.argmax(dim=1)
                pred_earring = logits_earring.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                # total_correct += torch.eq(pred, label).float().sum().item()
                total_correct_hair += torch.eq(pred_hair, label[:, 0]).float().sum().item()
                total_correct_hair_color += torch.eq(pred_hair_color, label[:, 1]).float().sum().item()
                total_correct_gender += torch.eq(pred_gender, label[:, 2]).float().sum().item()
                total_correct_earring += torch.eq(pred_earring, label[:, 3]).float().sum().item()
                total_num += x.size(0)
            # test_acc = total_correct / total_num
            test_acc1 = total_correct_hair / total_num
            test_acc2 = total_correct_hair_color / total_num
            test_acc3 = total_correct_gender / total_num
            test_acc4 = total_correct_earring / total_num
            # print("epoch:", epoch, " acc:", test_acc)
            print("epoch:", epoch, " acc1:", test_acc1, " acc2:", test_acc2, " acc3:", test_acc3, " acc4:", test_acc4)

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), model_path)
        #     print('saving model with acc {:.3f}'.format(best_acc))
        if test_acc2 > best_acc2:
            best_acc1 = test_acc1
            best_acc2 = test_acc2
            best_acc3 = test_acc3
            best_acc4 = test_acc4
            torch.save(model.state_dict(), model_path)
            print('saving model with acc1 ', round(best_acc1, 3), ' acc2', round(best_acc2, 3), ' acc3', round(best_acc3, 3), ' acc4', round(best_acc4, 3))

    print('best acc1:', round(best_acc1, 3), ' best acc2:', round(best_acc2, 3), ' acc3', round(best_acc3, 3), ' acc4', round(best_acc4, 3))


if __name__ == '__main__':
    main()