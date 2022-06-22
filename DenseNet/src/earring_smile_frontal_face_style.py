import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from data import FS2KData

import torchvision.models as models
from model import ModelDenseNet

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
import config


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


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

    def __init__(self, num_class, alpha=[0.1825, 0.15, 0.235, 0.2375, 0.195], gamma=2, balance_index=-1, smooth=None,
                 size_average=True):
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


# 获取混淆矩阵
def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    """
    Calculate confusion matrix on the provided preds and labels.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
        num_classes (int): number of classes. Tensor(s) must be on CPU.
        normalize (Optional[str]) : {‘true’, ‘pred’, ‘all’}, default="true"
            Normalizes confusion matrix over the true (rows), predicted (columns)
            conditions or all the population. If None, confusion matrix
            will not be normalized.
    Returns:
        cmtx (ndarray): confusion matrix of size (num_classes x num_classes)
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    return cmtx


# 绘制混淆矩阵
def plot_confusion_matrix(cmtx, num_classes, filename, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig('../image/confusion_matrix/' + filename + '.png')
    return figure


# 把混淆矩阵加到tensorboard里
def add_confusion_matrix(
    writer,
    cmtx,
    num_classes,
    global_step=None,
    subset_ids=None,
    class_names=None,
    tag="Confusion Matrix",
    figsize=None,
):
    """
    Calculate and plot confusion matrix to a SummaryWriter.
    Args:
        writer (SummaryWriter): the SummaryWriter to write the matrix to.
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        global_step (Optional[int]): current step.
        subset_ids (list of ints): a list of label indices to keep.
        class_names (list of strs, optional): a list of all class names.
        tag (str or list of strs): name(s) of the confusion matrix image.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    """
    if subset_ids is None or len(subset_ids) != 0:
        # If class names are not provided, use class indices as class names.
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        # If subset is not provided, take every classes.
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            class_names=sub_names,
            figsize=figsize,
            filename=tag.split(' ')[len(tag.split(' ')) - 1]
        )
        # Add the confusion matrix image to writer.
        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)


def get_criterion(num_classes, device):
    if num_classes == 2:
        return nn.CrossEntropyLoss().to(device)
    return FocalLoss(num_class=num_classes, alpha=config.focal_loss_alpha_dict[num_classes])


def main():
    batch_size = config.batch_size

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

    file_path_train = config.file_path_train
    file_path_test = config.file_path_test
    json_file_train = config.json_file_train
    json_file_test = config.json_file_test

    train_set = FS2KData(json_path=json_file_train,
                         img_path=file_path_train,
                         mode='train',
                         feature_list=config.feature_train)

    cifar_train = DataLoader(train_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True,
                             pin_memory=True, )

    test_set = FS2KData(json_path=json_file_test,
                        img_path=file_path_test,
                        mode='test',
                        feature_list=config.feature_train)
    cifar_test = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            drop_last=False,
                            pin_memory=True, )

    x, label = iter(cifar_train).next()  # iter得到DataLoader的迭代器 再用迭代器的next方法得到一个batch

    print("x: ", x.shape, "label: ", label.shape)

    device = torch.device('cuda')
    # model = ResNet18()
    # model = res_model(2)
    # model = ModelDenseNet()
    model = ModelDenseNet()
    model_path = config.model_path
    if torch.cuda.device_count() > 1:  # 查看当前电脑的可用的gpu的数量，若gpu数量>1,就多gpu训练
        model = torch.nn.DataParallel(model)  # 多gpu训练,自动选择gpu
    #
    model.to(device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # criteon = nn.CrossEntropyLoss().to(device)
    # criteon3 = FocalLoss(num_class=3, alpha=config.focal_loss_alpha_dict[3])
    optimizer = optim.Adam(model.parameters(), lr=config.lr_base)  # 定义一个优化器
    scheduler = None
    # 使用学习率衰减
    if config.lr_mode == 'Exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    elif config.lr_mode == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.lr_t_max, eta_min=config.lr_eta_min, last_epoch=config.lr_last_epoch)



    print(model)  # 打印出网络结构

    best_acc = 0.0
    best_acc1 = 0.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    best_acc4 = 0.0
    for epoch in range(config.epoch):
        model.train()
        # 预测值和标注值，用于绘制混淆矩阵
        preds_0 = []
        preds_1 = []
        preds_2 = []
        preds_3 = []
        labels = []
        for batchidx, (x, label) in enumerate(tqdm(cifar_train)):# enumerate可以生成类似元组 既得到元素又得到索引
            # x: [b, 3, 32, 32]
            # label: [b]
            x, label = x.to(device), label.to(device)
            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar 长度为0的标量
            logits_1, logits_2, logits_3, logits_4 = model(x)  # 调用Lenet5的forward方法

            loss_1 = get_criterion(2, device)(logits_1, label[:, 0].long())
            loss_2 = get_criterion(2, device)(logits_2, label[:, 1].long())
            loss_3 = get_criterion(2, device)(logits_3, label[:, 2].long())
            loss_4 = get_criterion(3, device)(logits_4, label[:, 3].long())
            loss = loss_1 + loss_2 * 0.5 + loss_3 * 0.3 + loss_4 * 0.25
            try:

                # 需将tensor从gpu转到cpu上
                preds_0.append(logits_1.cpu())
                preds_1.append(logits_2.cpu())
                preds_2.append(logits_3.cpu())
                preds_3.append(logits_4.cpu())
                labels.append(label.cpu())
                # backprop
                optimizer.zero_grad()# 梯度清零
                # 后向的时候是做梯度累加 所以需要先清零
                loss.backward()
                optimizer.step()
                if config.lr_mode != 'off':
                    scheduler.step()
            except:
                continue


        #
        # print("epoch:", epoch, " loss:", loss.item())# item将标量转化为tensor
        # print("epoch:", epoch, " loss_hair:", loss_hair.item(), " loss_hair_color:", loss_hair_color.item(), " loss_gender:", loss_gender.item())
        print("epoch:", epoch, " loss_earring", loss_1.item(), " loss_smile: ", loss_2.item(), " loss_frontal_face: ", loss_3.item(), " loss_style: ", loss_4.item())

        model.eval()
        with torch.no_grad():# 告诉pytorch不需要后向传播
            #test
            # total_correct = 0
            total_correct_1 = 0
            total_correct_2 = 0
            total_correct_3 = 0
            total_correct_4 = 0
            total_num = 0
            for x, label in tqdm(cifar_test):

                x, label = x.to(device), label.to(device)

                # logits = model(x)
                logits_1, logits_2, logits_3, logits_4 = model(x)
                # [b]
                # pred = logits.argmax(dim=1)
                pred_1 = logits_1.argmax(dim=1)
                pred_2 = logits_2.argmax(dim=1)
                pred_3 = logits_3.argmax(dim=1)
                pred_4 = logits_4.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                # total_correct += torch.eq(pred, label).float().sum().item()
                total_correct_1 += torch.eq(pred_1, label[:, 0]).float().sum().item()
                total_correct_2 += torch.eq(pred_2, label[:, 1]).float().sum().item()
                total_correct_3 += torch.eq(pred_3, label[:, 2]).float().sum().item()
                total_correct_4 += torch.eq(pred_4, label[:, 3]).float().sum().item()
                total_num += x.size(0)
            # test_acc = total_correct / total_num
            test_acc1 = total_correct_1 / total_num
            test_acc2 = total_correct_2 / total_num
            test_acc3 = total_correct_3 / total_num
            test_acc4 = total_correct_4 / total_num
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
            print('saving model with acc1 ', round(best_acc1, 3), ' acc2 ', round(best_acc2, 3), ' acc3 ', round(best_acc3, 3), ' acc4 ', round(best_acc4, 3))

        # tensorboard
        mAP = (best_acc1 + best_acc2 + best_acc3 + best_acc4) / 4
        ap_1 = best_acc1
        ap_2 = best_acc2
        ap_3 = best_acc3
        ap_4 = best_acc4
        writer = SummaryWriter('../log')
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('AP_earring', ap_1, epoch)
        writer.add_scalar('AP_smile', ap_2, epoch)
        writer.add_scalar('AP_frontal_face', ap_3, epoch)
        writer.add_scalar('AP_style', ap_4, epoch)

        """
        保存训练情况，输出成图片
        """
        plt.plot(epoch, mAP, label='mAP')
        plt.savefig('../image/train/mAP.png')
        plt.plot(epoch, ap_1, label='AP_earring')
        plt.savefig('../image/train/AP_earring.png')
        plt.plot(epoch, ap_2, label='AP_smile')
        plt.savefig('../image/train/AP_smile.png')
        plt.plot(epoch, ap_3, label='AP_frontal_face')
        plt.savefig('../image/train/AP_frontal_face.png')
        plt.plot(epoch, ap_4, label='AP_style')
        plt.savefig('../image/train/AP_style.png')
        """
        混淆矩阵可视化
        """
        preds_0 = torch.cat(preds_0, dim=0)
        preds_1 = torch.cat(preds_1, dim=0)
        preds_2 = torch.cat(preds_2, dim=0)
        preds_3 = torch.cat(preds_3, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtxa = get_confusion_matrix(preds_0, labels, 2)
        cmtxb = get_confusion_matrix(preds_1, labels, 2)
        cmtxc = get_confusion_matrix(preds_2, labels, 2)
        cmtxd = get_confusion_matrix(preds_3, labels, 3)
        """
        保存混淆矩阵
        """

        add_confusion_matrix(writer=writer, cmtx=cmtxa, num_classes=2, class_names=[0, 1],
                             tag="Train Confusion Matrix for earring", figsize=[6.4, 4.8])
        add_confusion_matrix(writer=writer, cmtx=cmtxb, num_classes=2, class_names=[0, 1],
                             tag="Train Confusion Matrix for smile", figsize=[6.4, 4.8])
        add_confusion_matrix(writer=writer, cmtx=cmtxc, num_classes=2, class_names=[0, 1],
                             tag="Train Confusion Matrix for frontal_face", figsize=[6.4, 4.8])
        add_confusion_matrix(writer=writer, cmtx=cmtxd, num_classes=3, class_names=[0, 1, 2],
                             tag="Train Confusion Matrix for style", figsize=[6.4, 4.8])

    print('best acc1:', round(best_acc1, 3), ' best acc2:', round(best_acc2, 3), ' acc3:', round(best_acc3, 3), ' best acc4:')


if __name__ == '__main__':
    main()