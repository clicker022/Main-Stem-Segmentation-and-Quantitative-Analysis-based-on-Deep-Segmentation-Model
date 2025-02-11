import os
import cv2
import yaml
import numpy as np
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config
from PIL import Image
from mrcnn import visualize
from samples.coco import coco
import tensorflow as tf
ROOT_DIR = os.getcwd()
# 你的验证数据集路径
VALIDATION_DATASET_DIR = "D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\plant2227_dataset\\val_data"

# 你的模型文件夹路径
MODEL_DIR = "D:\All_Codes\MaskRCNNTest\Mask_RCNN\logs\shapes20230331T0050"


def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou


class InferenceConfig(Config):
    NAME = "shapes"
    NUM_CLASSES = 1 + 1  # background + 1 shapes
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels


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


            mask_path = mask_floder + "/" + filestr + ".png"

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
        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object


        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)


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



config = InferenceConfig()

config.display()


val_data_path = "samples/plant2227_dataset/val_data/"
val_img_floder = val_data_path + "pic"
val_mask_floder = val_data_path + "cv2_mask"
val_imglist = os.listdir(val_img_floder)
val_count = len(val_imglist)
# 加载验证数据集
dataset_val = DrugDataset()
dataset_val.vload_shapes(val_count, val_img_floder, val_mask_floder, val_imglist, val_data_path)
dataset_val.prepare()

# 初始化模型
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# 评估所有模型
best_model_path = None
best_iou = 0

model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]

for model_file in model_files:
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"Loading model from: {model_path}")

    # 加载权重
    model.load_weights(model_path, by_name=True)

    # 评估模型
    image_ids = np.random.choice(dataset_val.image_ids, 50)  # 可以更改为验证集中的所有图像
    IOUs = []

    for image_id in image_ids:
        # 加载图像和ground truth数据
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config, image_id,
                                                                                  use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

        # 进行预测
        results = model.detect([image], verbose=0)
        r = results[0]

        if len(r['class_ids']) > 0:
            # 获取第一个类别的mask
            pred_mask = r['masks'][:, :, 0]
            gt_mask = gt_mask[:, :, 0]

            # 计算IOU
            iou = compute_iou(gt_mask, pred_mask)
            IOUs.append(iou)

    # 计算平均IOU
    mean_iou = np.mean(IOUs)
    print(f"Model {model_path} mean IOU: {mean_iou}")

    # 更新最佳模型
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_model_path = model_path

print(f"Best model: {best_model_path} with mean IOU: {best_iou}")
# Best model: D:\All_Codes\MaskRCNNTest\Mask_RCNN\logs\shapes20230328T2343\mask_rcnn_shapes_0062.h5 with mean IOU: 0.723622643243299