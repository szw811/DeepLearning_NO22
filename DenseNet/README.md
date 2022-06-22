# 深度学习编程作业
## 介绍
感谢群友给我打的样，原群友版本的作业使用的是ResNet

本作业使用了DenseNet121，你也可以使用其他模型训练
## 目录结构
```
.
├── FS2K    数据集
├── image   图像存档
├── log     tensorboard存档
├── model   权重
├── src     源代码
```
## 训练
运行`main.py`
注意，`batch_size`位于`main.py`的第`416`行

我运行的情况是，当`batch_size=64`的时候
显存占用为`15161MiB / 24576MiB`

敬请依据显存大小调整batch大小
## 看板
运行`tensorboard.sh`

或者，如果你是windows系统，请直接在本目录下运行
```shell
tensorboard --logdir=./log
```
## 训练情况
本人测试，在`epoch=34`后，模型收敛