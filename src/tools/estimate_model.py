import cv2
import numpy as np
import os
import glob

def calculate_confusion_matrix(gt_mask, pred_mask):
    TP = np.sum((gt_mask == 1) & (pred_mask == 1))
    TN = np.sum((gt_mask == 0) & (pred_mask == 0))
    FP = np.sum((gt_mask == 0) & (pred_mask == 1))
    FN = np.sum((gt_mask == 1) & (pred_mask == 0))
    return TP, TN, FP, FN

def calculate_acc(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def calculate_iou(TP, FP, FN):
    return TP / (TP + FP + FN)

def compute_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def print_each(gt_mask_file, pred_mask_file, index, acc, iou):
    print("------------------------" + str(index) + "------------------------")
    print("gt_mask: " + os.path.basename(gt_mask_file))
    print("pred_mask: " + os.path.basename(pred_mask_file))
    print("Acc: " + str(acc))
    print("IoU: " + str(iou))


gt_mask_files = sorted(glob.glob("D:/All_Codes/MaskRCNNTest/Mask_RCNN/samples/plant2227_dataset/test_data/cv2_mask/*.png"))
pred_mask_files = sorted(glob.glob("D:/All_Codes/MaskRCNNTest/Mask_RCNN/samples/plant2227_dataset/test_data/test_results/r95/*.jpg"))

iou_list = []
acc_list = []
index = 0
for gt_mask_file, pred_mask_file in zip(gt_mask_files, pred_mask_files):
    gt_mask = cv2.imread(gt_mask_file)
    pred_mask = cv2.imread(pred_mask_file)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
    gt_, gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pred_, pred_mask = cv2.threshold(pred_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Threshold the masks if necessary
    gt_mask[gt_mask > 0] = 1
    pred_mask[pred_mask > 0] = 1
    # gt_mask = np.logical_not(gt_mask).astype(np.uint8)
    # pred_mask = np.logical_not(pred_mask).astype(np.uint8)

    # gt_mask = gt_mask.astype(np.uint8)
    # pred_mask = pred_mask.astype(np.uint8)
    # cv2.imshow("gt",gt_mask*255)
    # cv2.imshow("pr",pred_mask*255)
    # cv2.waitKey(0)

    TP, TN, FP, FN = calculate_confusion_matrix(gt_mask, pred_mask)
    acc = calculate_acc(TP, TN, FP, FN)
    iou = calculate_iou(TP, FP, FN)

    index = index + 1
    print_each(gt_mask_file, pred_mask_file, index, acc, iou)

    acc_list.append(acc)
    iou_list.append(iou)

mAcc = np.mean(acc_list)
mIoU = np.mean(iou_list)

print("Mean Accuracy (mAcc):", mAcc)
print("Mean Intersection over Union (mIoU):", mIoU)
