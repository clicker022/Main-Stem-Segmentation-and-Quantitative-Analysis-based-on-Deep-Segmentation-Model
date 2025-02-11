# -*- coding: utf-8 -*-
import os
import random
import sys
import skimage.io
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from datetime import datetime
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory of save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "MaskRCNNTest\\Mask_RCNN\\logs")

# Local path of trained weights file
MODEL_PATH = os.path.join(MODEL_DIR, "shapes20230331T0050\\mask_rcnn_shapes_0095.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\pic")

SAVE_DIR = os.path.join(ROOT_DIR, "MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\test_results\\r95")


class ShapesConfig(Config):
    NAME = "shapes"
    NUM_CLASSES = 1 + 1  # background + 1 shapes
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(MODEL_PATH, by_name=True)
# model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
class_names = ['BG', ""]  # 修改显示在图像中的类别名称----------------------------------------------------------------------

"""随机预测一张图像

file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

a = datetime.now()
results = model.detect([image], verbose=1)
b = datetime.now()
print("time:", (b - a).seconds, " seconds")
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
"""

# 以下代码为批量预测图像并去除白色背景填充,然后保存

# 将visualize.py的display_instances()复制过来重写
def display_instances(count,image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)   #显示图像的外围边框，保存原始图像大小的数据时需要注释掉
    # ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        new_image = np.zeros_like(image)
        mask = masks[:, :, i]
        if show_mask:
            # 修改alpha改变mask颜色透明度,将masked_image改为new_image，使背景颜色为黑色----------------------------------------
            masked_image = visualize.apply_mask(new_image, mask, color, alpha=1)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)  # 将edgecolor的color改为”none“，去除mask边缘线----------
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        # 保存预测结果图像

        fig = plt.gcf()
        fig.set_size_inches(width / 100.0, height / 100.0)  # 输出原始图像width*height的像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(SAVE_DIR + "/%s.jpg" % (str(count[0:11])), pad_inches=0.0)  # 使用原始图像名字的第1到第11个字符来命名新的图像

        # 保存预测结果图像
        # plt.show()                    #在保存预测结果图像时，如果不想每保存一张显示一次，可以把他注释掉


# 批量预测
count = os.listdir(IMAGE_DIR)
custom_color = [[1,1,1],[0,0,0]]
for i in range(0, len(count)):
    path = os.path.join(IMAGE_DIR, count[i])
    if os.path.isfile(path):
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        # display_instances(count[i], image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
        display_instances(count[i], image, r['rois'], r['masks'], r['class_ids'], class_names,
                          show_bbox=None, colors=custom_color)
