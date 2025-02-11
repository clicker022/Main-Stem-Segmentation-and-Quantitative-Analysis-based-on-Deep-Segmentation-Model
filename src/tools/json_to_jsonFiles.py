#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

if __name__ == '__main__':
    json_dir = "D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\plant2227_dataset\json"  # 存放labelme标注后的json文件
    for name in os.listdir(json_dir):
       # print(name)
        file = os.path.splitext(name)
        filename, filetype = file
        if filetype == '.json':
            json_path = os.path.join(json_dir, name)
            os.system(str("labelme_json_to_dataset " + json_path))
            print("success json to dataset: ", json_path)