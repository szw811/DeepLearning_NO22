import cv2
import json

file_path_train = '../FS2K/train/photo'
json_file_train = '../FS2K/anno_train.json'
img_path = '../FS2K/copy_img'

f = open(json_file_train, 'r', encoding='utf-8')
json_data = json.load(f)
# print(json_data)


for i in range(len(json_data)):
    if (json_data[i]['hair_color'] == 2):
        name = json_data[i]['image_name'].replace('/image', '_image')
        name1 = file_path_train + '/' + name + '.png'
        img = cv2.imread(name1)
        h_flip = cv2.flip(img, 1)  # 水平翻转
        cv2.imwrite(img_path + '/' + name + '2.png', h_flip, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        json_data[i]['image_name'] = name + '2'
        json_data.append(json_data[i])
    if (json_data[i]['hair_color'] == 3):
        name = json_data[i]['image_name'].replace('/image', '_image')
        name1 = file_path_train + '/' + name + '.png'
        img = cv2.imread(name1)
        h_flip = cv2.flip(img, 1)  # 水平翻转
        cv2.imwrite(img_path + '/' + name + '2.png', h_flip, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        json_data[i]['image_name'] = name + '2'
        json_data.append(json_data[i])

with open("./new_json.json", 'w') as write_f:
	write_f.write(json.dumps(json_data, indent=4, ensure_ascii=False))




# v_flip = cv2.flip(img, 0)  # 垂直翻转
# hv_flip = cv2.flip(img, -1)  # 水平垂直翻转




