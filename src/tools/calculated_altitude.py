import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from skimage import morphology
import math


def sort_box_vertices(box):
    """
    Sort the vertices of a rectangle in clockwise order
    This function calculates the angle between each vertex of the rectangle and the center point,
    then sorts the vertices based on their angle. The result is a new list of vertices with the
    vertices sorted in clockwise order.
    """
    # find the center of the box
    center = np.mean(box, axis=0)

    # calculate the angle between each vertex and the center point
    angles_i = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])

    # sort the vertices based on their angle
    sorted_idx = np.argsort(angles_i)

    # return the sorted vertices
    return box[sorted_idx]


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return angle


def find_nearest_90_degree_angle(sorted_box):
    min_diff = 90
    nearest_90_degree_index = 0

    for i in range(4):
        p1 = sorted_box[i]
        p2 = sorted_box[(i + 1) % 4]
        p3 = sorted_box[(i + 2) % 4]

        vec1 = p1 - p2
        vec2 = p3 - p2

        angle = angle_between(vec1, vec2)

        diff = abs(angle - 90)
        if diff < min_diff:
            min_diff = diff
            nearest_90_degree_index = i

    return nearest_90_degree_index, sorted_box[nearest_90_degree_index], sorted_box[(nearest_90_degree_index + 1) % 4], sorted_box[(nearest_90_degree_index + 2) % 4]


def calculate_soybean_height_mm(refer_mask_image, soybean_mask_image, ref_height_mm, ref_width_mm):
    """
    first convert the mask image to grayscale and threshold it to obtain a binary image.
    We then use the cv2.findContours() function to detect contours in the binary image.
    We set the RETR_EXTERNAL flag to retrieve only the outermost contours, and the CHAIN_APPROX_SIMPLE flag to
    retrieve only the endpoints of the contours.
    Next, we loop through the contours and use the cv2.approxPolyDP() function to approximate each contour with a
    polygon.If the polygon has exactly four vertices, we assume it is a rectangular reference object and add it to the
    rects list.
    Note that the approxPolyDP() function approximates a contour with a polygon with fewer vertices,
    but the approximation accuracy can be controlled by the second argument. In this example code,
    we set the second argument to 0.1 times the perimeter of the contour, which should provide a reasonably accurate
    approximation for rectangular objects. However, you may need to adjust this value depending on your specific
    dataset. the rectangular mask of the reference object is not a standard rectangle but instead has a tilt angle,
    we need to adjust the method used to calculate the pixel height (ref_height_px).
    we use the minimum bounding rectangle (MBR) of the reference object mask to estimate its height.
    The MBR is the smallest rectangle that can fully enclose the object mask, regardless of its orientation or shape
        find the minimum circumscribed rectangle of the soybean mask in the soybean plant mask picture
    and calculate the pixel height of the minimum circumscribed rectangle
    """
    # convert the mask image to grayscale
    refer_gray = cv2.cvtColor(refer_mask_image, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image to obtain a binary image
    refer_, refer_thresh = cv2.threshold(refer_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find contours in the binary image
    refer_contours, refer_ = cv2.findContours(refer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop through the contours and filter out those that are not rectangular
    refer_rects = []
    for refer_contour in refer_contours:
        perimeter = cv2.arcLength(refer_contour, True)
        approx = cv2.approxPolyDP(refer_contour, 0.1 * perimeter, True)
        if len(approx) == 4:
            refer_rects.append(approx.reshape(-1, 2))

    # Loop through each of the rectangular reference objects and calculate their aspect ratio and angle.
    aspect_ratios = []
    sorted_boxes = []
    for refer_rect in refer_rects:
        # rect contains the coordinates of the rectangular reference object

        # get the minimum bounding rectangle of the reference object mask
        rect = cv2.minAreaRect(refer_rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # sort the vertices of the MBR in clockwise order
        sorted_box = sort_box_vertices(box)
        sorted_boxes.append(sorted_box)

        # 查找夹角最接近90度的邻边
        index, p1, p2, p3 = find_nearest_90_degree_angle(sorted_box)
        w = euclidean_distance(p1, p2)
        h = euclidean_distance(p2, p3)
        if w > h:
            temp = h
            h = w
            w = temp
        aspect_ratios.append(w / h)

    # Choose the reference object with the closest aspect ratio to the expected aspect ratio of the reference object.
    expected_aspect_ratio = ref_width_mm / ref_height_mm
    best_rect_idx = np.argmin(np.abs(np.array(aspect_ratios) - expected_aspect_ratio))
    best_rect = sorted_boxes[best_rect_idx]

    # calculate the height of the reference object in the image, calculate the pixel height of the MBR

    indexb, p1b, p2b, p3b = find_nearest_90_degree_angle(best_rect)
    ref_width_px = euclidean_distance(p1b, p2b)
    ref_height_px = euclidean_distance(p2b, p3b)
    if ref_width_px > ref_height_px:
        ref_temp = ref_height_px
        ref_height_px = ref_width_px
        ref_width_px = ref_temp

    # convert the soybean mask to grayscale
    soybean_gray = cv2.cvtColor(soybean_mask_image, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image to obtain a binary image
    soybean_, soybean_thresh = cv2.threshold(soybean_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform a morphological closing operation to connect the regions
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(soybean_thresh)

    """
    # Loop until there is only one connected component
    kernel_size = 4
    while num_labels > 2:
        # Perform connected component closure
        kernel_size = kernel_size + 1
        c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        soybean_thresh = cv2.morphologyEx(soybean_thresh, cv2.MORPH_CLOSE, c_kernel)
        # Find connected components again
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(soybean_thresh)
    """

    # find contours in the binary image
    soybean_contours, soybean_ = cv2.findContours(soybean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the largest contour
    ## soybean_contours_sorted = sorted(soybean_contours, key=cv2.contourArea, reverse=True)[:1]

    # find the minimum bounding rectangle of the soybean mask
    ## soybean_rect = cv2.minAreaRect(soybean_contours_sorted[0])


    merged_contours = np.vstack(soybean_contours)
    soybean_rect = cv2.minAreaRect(merged_contours)


    # get the minimum bounding rectangle of the soybean mask
    soybean_box = cv2.boxPoints(soybean_rect)
    soybean_box = np.int0(soybean_box)

    # sort the vertices of the MBR in clockwise order
    soybean_sorted_box = sort_box_vertices(soybean_box)

    # calculate the pixel height of the minimum bounding rectangle
    soybean_width_px = euclidean_distance(soybean_sorted_box[1][1], soybean_sorted_box[0][1])
    soybean_height_px = euclidean_distance(soybean_sorted_box[3][1], soybean_sorted_box[0][1])
    if soybean_width_px > soybean_height_px:
        soybean_temp = soybean_height_px
        soybean_height_px = soybean_width_px
        soybean_width_px = soybean_temp

    # calculate the height of the soybean plant ''soybean_height_px = soybean_rect[3][1] - soybean_rect[0][1]
    soybean_height_mm = (ref_height_mm * soybean_height_px) / ref_height_px

    return soybean_height_mm


REFER_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\ruler_results")
SOYBEAN_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\test_results\\r95")
# SOYBEAN_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\cv2_mask")
SOURCE_IMAGE_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\pic")
TRUE_SOYBEAN_HEIGHT_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\true_length\\")
SAVE_PATH = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\temp")
ref_width_mm = 25   # width of the reference object in reality in mm /*30
ref_height_mm = 28  # height of the reference object in reality in mm /*35
ref2_width_mm = 30   # width of the reference object in reality in mm /*30
ref2_height_mm = 35  # height of the reference object in reality in mm /*35
threshold = 80

n = 222
count = os.listdir(SOYBEAN_MASK_DIR)
diff_list = []
diff_r_list = []
HAE = -1
HRE = -1
skip_values = ['Soybean0878', 'Soybean1871', 'Soybean2433']
for i in range(n):
    PIC_NAME = '%s' % str(count[i][0:11])
    if PIC_NAME in skip_values:
        continue
    refer_mask_image = cv2.imread(os.path.join(REFER_MASK_DIR, PIC_NAME + '.jpg'))
    soybean_mask_image = cv2.imread(os.path.join(SOYBEAN_MASK_DIR, PIC_NAME + '.jpg'))
    source_image = cv2.imread(os.path.join(SOURCE_IMAGE_DIR, PIC_NAME + '.jpg'))

    soybean_height_mm = calculate_soybean_height_mm(refer_mask_image, soybean_mask_image, ref_height_mm, ref_width_mm)

    with open(TRUE_SOYBEAN_HEIGHT_DIR + PIC_NAME + '.txt', 'r') as input_file:
        content = input_file.read().strip()
    try:
        soybean_true_height = int(content)
    except ValueError:
        try:
            soybean_true_height = float(content)
        except ValueError:
            print("文件内容无法转换为数字")

    diff = abs(soybean_height_mm - soybean_true_height)
    diff_r = diff / soybean_true_height

    # if diff > threshold:
    #    print('-------' + PIC_NAME + '-------')
    #    soybean_height_mm = calculate_soybean_height_mm(refer_mask_image, soybean_mask_image, ref2_height_mm, ref2_width_mm)
    #    diff = abs(soybean_height_mm - soybean_true_height)
    #    diff_r = diff / soybean_true_height

    diff_list.append(diff)
    diff_r_list.append(diff_r)

    if diff > HAE:
        HAE = diff
    if diff_r > HRE:
        HRE = diff_r
    print('-------' + PIC_NAME + '-------')
    print('true_soybean_height_mm = ', soybean_true_height)
    print('predict_soybean_height_mm = ', soybean_height_mm)


MAE = np.mean(diff_list)
MRE = np.mean(diff_r_list)

print("************************")
print("MAE = ", MAE)
print("MRE = ", MRE)
print("HAE = ", HAE)
print("HRE = ", HRE)
