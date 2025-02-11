# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image
from datetime import datetime
from queue import Queue
from multiprocessing import Pool
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

num_iterations = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs\\shapes20230319T2328\\mask_rcnn_shapes_0789.h5")

# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


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
    # IMAGE_MIN_DIM = 256
    # IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 10
    # MAX_GT_INSTANCES = 100

    """
    The "steps_per_epoch" and "validation_steps" parameters in Keras specify the number of batches of samples to use in
     each epoch during training and validation, respectively.

    To determine their values, divide the total number of samples in your training set by the batch size. For example, if 
    you have 1000 training samples and a batch size of 20, then steps_per_epoch = 1000 / 20 = 50.

    For validation, the same formula applies, but with the number of samples in your validation set. For example, if you 
    have 200 validation samples and a batch size of 20, then validation_steps = 200 / 20 = 10.

    It's important to choose values for steps_per_epoch and validation_steps that ensure that all the samples in your 
    training set and validation set are used in each epoch, without exceeding memory constraints.
    """
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1783

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 222


config = ShapesConfig()
config.display()


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
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
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
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]

            # Step1:使用.npz格式的mask文件时，将这行注释掉：
            # mask_path = mask_floder + "/" + filestr + ".png"
            # 改为：
            mask_path = ROOT_DIR + "/resources/shapes/rwmask/train_mask/" + filestr + ".npz"

            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            print(type(cv_img))

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)


    def vload_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
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
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]

            # Step1:使用.npz格式的mask文件时，将这行注释掉：
            # mask_path = mask_floder + "/" + filestr + ".png"
            # 改为：
            mask_path = ROOT_DIR + "/resources/shapes/rwmask/val_mask/" + filestr + ".npz"

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
        global num_iterations
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object

        # Step2:使用.npz格式的mask文件时，注释掉这里
        """
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        """

        # Step3:使用.npz格式的mask文件时，增加下面这行：
        mask = np.load(info['mask_path'])["arr_0"]

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("main_stem") != -1:
                labels_form.append("main_stem")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def cosine_annealing_schedule(epoch, lr):
    total_epochs = 100
    max_lr = 0.001
    min_lr = 1e-6

    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def train_model():
    # 基础设置
    # dataset_root_path = "samples/plant2227_dataset/"  # 你的数据的路径
    train_data_path = "samples/plant2227_dataset/train_data/"
    val_data_path = "samples/plant2227_dataset/val_data/"

    train_img_floder = train_data_path + "pic"
    train_mask_floder = train_data_path + "cv2_mask"

    val_img_floder = val_data_path + "pic"
    val_mask_floder = val_data_path + "cv2_mask"
    # yaml_floder = dataset_root_path

    train_imglist = os.listdir(train_img_floder)
    val_imglist = os.listdir(val_img_floder)

    train_count = len(train_imglist)
    val_count = len(val_imglist)

    # train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(train_count, train_img_floder, train_mask_floder, train_imglist, train_data_path)
    dataset_train.prepare()

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    dataset_val.vload_shapes(val_count, val_img_floder, val_mask_floder, val_imglist, val_data_path)
    dataset_val.prepare()

    config = ShapesConfig()
    config.display()


    # print("dataset_val-->",dataset_val._image_ids)

    # Load and display random samples
    # image_ids = np.random.choice(dataset_train.image_ids, 4)
    # for image_id in image_ids:
    #    image = dataset_train.load_image(image_id)
    #    mask, class_ids = dataset_train.load_mask(image_id)
    #    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "last"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # 添加该学习率调度器，动态调整学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
    lr_scheduler = LearningRateScheduler(cosine_annealing_schedule, verbose=1)
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')
                # ,custom_callbacks=reduce_lr)  # 固定其他层，只训练head，epoch为50

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers="all",
                custom_callbacks=[lr_scheduler])  # 微调所有层的参数，epoch前50为(lr=0.0001)，50-100为(lr=0.00001)


class PlantConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def predict():
    import skimage.io
    from mrcnn import visualize

    # Create models in training mode
    config = PlantConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model_path = model.find_last()
    # model_path_test = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\logs\\shapesxxxx\\mask_rcnn_shapes_xxxx.h5")
    # Load trained weights (fill in path to trained weights here)
    # assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    # model.load_weights(model_path, by_name=True,
    #                  exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    class_names = ['BG', 'main_stem']

    # Load a random image from the images folder
    # file_names = r'D:\All_Codes\MaskRCNNTest\Mask_RCNN\plantsImages\Soybean3815.jpg'
    # image = skimage.io.imread(file_names)
    IMAGE_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\plantsImages_testSet")
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))


    # Run detection
    a = datetime.now()
    results = model.detect([image], verbose=1)
    b = datetime.now()
    print("time:", (b - a).seconds, " seconds")

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == "__main__":
    train_model()
    # predict()
    # 查看loss曲线图：tensorboard --logdir=logs

