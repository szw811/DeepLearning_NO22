# import torchvision.models as models
# from FS2K_dataset import FS2KData as D
# import torch.nn as nn
# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader
#
# def confusion_matrix(preds, labels, conf_matrix):
#     preds = torch.argmax(preds, 1)
#     for p, t in zip(preds, labels):
#         conf_matrix[p, t] += 1
#     return conf_matrix
#
# def get_device():
#     return 'cuda' if torch.cuda.is_available() else 'cpu'
#
#
# device = get_device()
#
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         model = model
#         for param in model.parameters():
#             param.requires_grad = False
#
#
# def res_model(num_classes, feature_extract=False, use_pretrained=True):
#     model_ft = models.resnet34(pretrained=use_pretrained)
#     set_parameter_requires_grad(model_ft, feature_extract)
#     num_ftrs = model_ft.fc.in_features
#     # 设置最后全连接层的输出
#     model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
#                                 nn.ReLU(),
#                                 nn.Linear(512, 128),
#                                 nn.ReLU(),
#                                 nn.Linear(128, 5),
#                                 nn.ReLU(),
#                                 )
#     return model_ft
#
# model = res_model(2)
# model_path = 'pre_res_model_earring.ckpt'
# # create model and load weights from checkpoint
# model = model.to(device)
# model.load_state_dict(torch.load(model_path))
#
# model.eval()
# predictions = []
#
# file_path_test = 'FS2K/test/photo'
# json_file_test = 'FS2K/anno_test.json'
#
# # 定义数据集
# test_set = D(json_file_test, file_path_test, 'test')
# test_loader = DataLoader(test_set,
#                          batch_size=16,
#                          shuffle=False,
#                          num_workers=0,
#                          drop_last=False,
#                          pin_memory=True,)
#
# labels = []
# conf_matrix = torch.zeros(2, 2)
# with torch.no_grad():
#     for step, (imgs, targets) in tqdm(enumerate(test_loader)):
#         # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
#         # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
#         targets = targets.squeeze()  # [50,1] ----->  [50]
#
#         # 将变量转为gpu
#         targets = targets.cuda()
#         imgs = imgs.cuda()
#         # print(step,imgs.shape,imgs.type(),targets.shape,targets.type())
#
#         out = model(imgs)
#         # 记录混淆矩阵参数
#         conf_matrix = confusion_matrix(out, targets, conf_matrix)
#         conf_matrix = conf_matrix.cpu()
#
# # for batch in tqdm(test_loader):
# #     imgs, label = batch
# #     with torch.no_grad():
# #         logits = model(imgs.to(device))
# #     predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
# #     labels.extend(label.cpu().numpy())
# #
# # print(predictions)
# # print(labels)
# # print(test_set.paths[0])
# # old_img_name = test_set.paths[0][16:]
# # new_img_name = old_img_name[:6] + '/' + old_img_name[7:-4]
# # print(new_img_name)
# # i = 0
# # while i < len(test_set.paths):
# #     if test_set.json_data[i]['image_name'] == new_img_name:
# #         hair_color = test_set.json_data[i]['hair_color']
# #         break
# #     i = i + 1
# # print(hair_color)