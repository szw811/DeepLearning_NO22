由于文件过多，模型过大无法上传github，可以从百度云链接下载数据集和预训练模型。</br>
运行时的主要文件的路径如下：
```
FS2K
├── FS2K
│       ├── test
│       └──train
├── confusion.py
├── FS2K_dataset.py
├── main_train.py
├── pre_res_model_gender.ckpt
├──...
```
res34_gender等文件夹记录了训练时的曲线变化，可通过tensorboard查看</br>
```
tenboard --logdir=res34_gender
```
</br>

如何训练
```
python main_train.py --attribute 人脸属性 --num_classes 属性类别数
```
以训练对性别进行分类的模型为例
```
python main_train.py --attribute ‘gender’ --num_classes 2
```
如何生成混淆矩阵
```
python confusion.py --attribute 人脸属性 --num_classes 属性类别数
```
以生成对性别进行分类的混淆矩阵为例
```
python confusion.py --attribute ‘gender’ --num_classes 2
```
[数据集和预训练模型的百度云下载链接](https://pan.baidu.com/s/1P3sTOrAlNTWRQfjed1sk8Q)提取码：95k9 


