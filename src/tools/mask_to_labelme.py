import os

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from labelme import utils
from json import dumps
import base64

def mask_to_polygons(mask, epsilon=0.8):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        cnt = cnt.squeeze(axis=1)  # (NumPoints, 1, 2) -> (NumPoints, 2)
        if cnt.size < 6:
            continue

        # simplified_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_cnt = simplified_cnt.reshape(-1, 2)

        polygons.append(simplified_cnt.tolist())

    return polygons

def image_to_base64(image_path):
    # 读取二进制图片，获得原始字节码
    with open(image_path, 'rb') as jpg_file:
        byte_content = jpg_file.read()

    # 把原始字节码编码成base64字节码
    base64_bytes = base64.b64encode(byte_content)

    # 把base64字节码解码成utf-8格式的字符串
    base64_string = base64_bytes.decode('utf-8')

    return base64_string



def main(source_image_path, mask_image_path, output_json_path):
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise FileNotFoundError(f"File not found: {mask_image_path}")

    mask_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)[1]
    polygons = mask_to_polygons(mask_image)

    # Create the LabelMe JSON structure
    data = {
        "version": "1.1",
        "flags": {},
        "shapes": [],
        "imagePath": str(Path(mask_image_path).name),
        "imageData": None,
    }

    for idx, poly in enumerate(polygons):
        data["shapes"].append({
            # "label": f"polygon_{idx}",
            "label": "main_stem",
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })

    # Encode the image as base64
    data["imageData"] = image_to_base64(source_image_path)

    # Save JSON data to file
    with open(output_json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    n = 5954
    count = os.listdir("D:\All_Codes\MaskRCNNTest\Mask_RCNN\mask_to_labelme\source")
    for i in range(n):
        source_image_path = 'D:\All_Codes\MaskRCNNTest\Mask_RCNN\mask_to_labelme\source/%s' % str(count[i][0:11]) + '.jpg'
        mask_image_path = 'D:\All_Codes\MaskRCNNTest\Mask_RCNN\mask_to_labelme\mask/%s' % str(count[i][0:11]) + '.jpg'
        output_json_path = 'D:\All_Codes\MaskRCNNTest\Mask_RCNN\mask_to_labelme\json/%s' % str(count[i][0:11]) + '.json'
        main(source_image_path, mask_image_path, output_json_path)
        print('%s' % str(count[i][0:11]) + '.json' + 'saved')


