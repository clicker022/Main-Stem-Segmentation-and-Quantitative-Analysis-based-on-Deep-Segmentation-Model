# -*- coding: utf-8 -*-
# 参考：https://blog.csdn.net/qq_35874169/article/details/112553195
# 将png形式的mask转换为npz格式的mask，以减少CPU读写占用，提高GPU使用率，提高训练速度
import os
import sys
import cv2
import yaml
import numpy as np
from PIL import Image
# import threading
from queue import Queue
from multiprocessing import Pool
from mrcnn import utils
from mrcnn.config import Config

ROOT_DIR = os.getcwd()


class ShapesConfig(Config):
    """Configuration for training on the shapes dataset.
    Derives from the base Config class and overrides values specific
    to the shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1

    # GPU显存不够的话调成1.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    # MAX_GT_INSTANCES = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 30


config = ShapesConfig()


class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.safe_load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        np.savez_compressed(os.path.join(ROOT_DIR, 'resources', 'shapes', 'rwmask', 'val_mask', info["path"].split("/")[-1].split(".")[0]), mask)
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "main_stem")

        for i in range(count):
            # 获取图片宽和高
            print(i)

            filestr = imglist[i].split(".")[0]

            # Step2：修改这里
            mask_path = mask_floder + "/" + filestr + ".png"
            # 改为：
            # mask_path = mask_floder + '\\rwmask\\' + filestr + '.npz'

            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            print(type(cv_img))

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object

        # Step3: 注释掉这里
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)

        # Step4: 加上这句
        # mask = np.load(info['mask_path'])["arr_0"]


def train_model():
    # 基础设置
    dataset_root_path = "samples/plant2227_dataset/val_data/"  # 你的数据的路径
    img_floder = dataset_root_path + "pic"

    # Step5：注释掉这句
    mask_floder = dataset_root_path + "cv2_mask"
    # 并改为：
    # mask_floder = os.path.join(ROOT_DIR, 'resources', 'shapes')

    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    count = len(imglist)


    # train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()

    config = ShapesConfig()
    config.display()

    # Step6:增加下列语句
    queue = Queue()
    thread_num = 30
    [queue.put(id) for id in dataset_train.image_ids]
    print("start! ")
    pthread = Pool(thread_num)

    while not queue.empty():
        image_id = queue.get()
        pthread.apply_async(dataset_train.load_mask, args=(image_id,))

    pthread.close()
    pthread.join()

    print("queue is empty!")




if __name__ == "__main__":
    train_model()
    # predict()