import cv2  # 计算机视觉库
from PIL import Image
import json  # 用来解析json文件
import numpy as np  # 科学计算包
import sys  # 处理Python运行时配置以及资源
import os  # 负责程序与操作系统的交互，提供了访问操作系统底层的接口。
import random  # 实现了各种分布的伪随机数生成
import time  # 处理时间的标准库
import base64  # 编解码库
from math import cos, sin, pi, fabs, radians  # 内置数学类函数库

images_path = 'D:/CollegeThings/diploma_project/dataset/data_enhancement/pic/'  # 图片的根目录
json_path = 'D:/CollegeThings/diploma_project/dataset/data_enhancement/json/'  # json文件的根目录
save_path = "D:/CollegeThings/diploma_project/dataset/data_enhancement/save/"  # 保存图片文件夹


# 读取json文件
def ReadJson(jsonfile):
    with open(jsonfile, encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData


# 保存json
def WriteJson(filePath, data):
    write_json = open(filePath, 'w')
    write_json.write(json.dumps(data, indent=2))
    write_json.close()


def rotate_bound(image, angle):
    h, w, _ = image.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    image_rotate = cv2.warpAffine(image, M, (nW, nH), borderMode=1)
    return image_rotate, cX, cY, angle


def RotateImage(img, degree):
    height, width = img.shape[:2]  # 获得图片的高和宽
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    print(width // 2, height // 2)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderMode=1)

    return imgRotation, matRotation


def rotate_xy(x, y, angle, cx, cy):
    # print(cx,cy)
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


# 转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code


# 坐标旋转
def rotatePoint(Srcimg_rotate, jsonTemp, M, imagePath):
    json_dict = {}
    for key, value in jsonTemp.items():
        if key == 'imageHeight':
            json_dict[key] = Srcimg_rotate.shape[0]
            print('Height ', json_dict[key])
        elif key == 'imageWidth':
            json_dict[key] = Srcimg_rotate.shape[1]
            print('Width ', json_dict[key])
        elif key == 'imageData':
            json_dict[key] = image_to_base64(Srcimg_rotate)
        elif key == 'imagePath':
            json_dict[key] = imagePath
        else:
            json_dict[key] = value
    for item in json_dict['shapes']:
        for key, value in item.items():
            if key == 'points':
                for item2 in range(len(value)):
                    pt1 = np.dot(M, np.array([[value[item2][0]], [value[item2][1]], [1]]))
                    value[item2][0], value[item2][1] = pt1[0][0], pt1[1][0]
    return json_dict


if __name__ == '__main__':
    file_list = os.listdir(images_path)
    i = 0
    for img_name in file_list:
        i = i + 1
        if i == 201:
            break
        SrcImg = cv2.imread(images_path + img_name)  # 读取图片
        JsonData = ReadJson(json_path + img_name[:-3] + 'json')  # 读取对应的json文件
        img_rotate, mat_rotate = RotateImage(SrcImg, -30)  # 旋转图片
        json_rotate = rotatePoint(img_rotate, JsonData, mat_rotate, img_name)
        cv2.imwrite(save_path + img_name[:-4] + "_r-30" + '.jpg', img_rotate)
        WriteJson(save_path + img_name[:-4] + "_r-30" + '.json', json_rotate)
        print(img_name, "is ok!")
