import os
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from skimage import morphology
import math

REFER_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\ruler_results")
# SOYBEAN_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\test_results\\r95")
SOYBEAN_MASK_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\cv2_mask")
SOURCE_IMAGE_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\pic")

SAVE_PATH = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\test_data\\temp")
PIC_NAME = 'Soybean0004'

refer_mask_image = cv2.imread(os.path.join(REFER_MASK_DIR, PIC_NAME + '.jpg'))
soybean_mask_image = cv2.imread(os.path.join(SOYBEAN_MASK_DIR, PIC_NAME + '.png'))
source_image = cv2.imread(os.path.join(SOURCE_IMAGE_DIR, PIC_NAME + '.jpg'))
ref_width_mm = 25   # width of the reference object in reality in mm /*30
ref_height_mm = 28  # height of the reference object in reality in mm /*35
new_image = np.zeros_like(source_image)


def transform_rect_points(sorted_box, M):
    rect_pixels = np.array([sorted_box[0], sorted_box[1], sorted_box[2], sorted_box[3]],
                           dtype=np.float32)  # 矩形参照物的最小外接矩形的四个顶点坐标

    # 使用cv2.perspectiveTransform计算透视变换后的顶点
    correct_points = cv2.perspectiveTransform(np.array([rect_pixels]), M)[0]
    correct_sorted_points = sort_box_vertices(correct_points)

    width_px = euclidean_distance(correct_sorted_points[1][1], correct_sorted_points[0][1])
    height_px = euclidean_distance(correct_sorted_points[3][1], correct_sorted_points[0][1])

    if width_px > height_px:
        temp = height_px
        height_px = width_px
        width_px = temp

    return correct_sorted_points, width_px, height_px


def correct_perspective(image, source_points, target_width, target_height, M = None):
    # 对源顶点进行排序
    # 按顺时针顺序排序顶点
    rect = np.zeros((4, 2), dtype="float32")

    s = source_points.sum(axis=1)
    rect[0] = source_points[np.argmin(s)]
    rect[2] = source_points[np.argmax(s)]

    diff = np.diff(source_points, axis=1)
    rect[1] = source_points[np.argmin(diff)]
    rect[3] = source_points[np.argmax(diff)]

    source_points_ordered = rect

    # 创建一个目标矩形，表示参照物在无透视形变情况下的顶点坐标
    target_points = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype="float32")

    if M is None:
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(source_points_ordered, target_points)

    # 应用透视变换矩阵
    warped = cv2.warpPerspective(image, M, (target_width, target_height))

    return warped, M


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
    # return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
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


"""
first convert the mask image to grayscale and threshold it to obtain a binary image.
We then use the cv2.findContours() function to detect contours in the binary image. 
We set the RETR_EXTERNAL flag to retrieve only the outermost contours, and the CHAIN_APPROX_SIMPLE flag to 
retrieve only the endpoints of the contours.

Next, we loop through the contours and use the cv2.approxPolyDP() function to approximate each contour with a polygon. 
If the polygon has exactly four vertices, we assume it is a rectangular reference object and add it to the rects list.

Note that the approxPolyDP() function approximates a contour with a polygon with fewer vertices, 
but the approximation accuracy can be controlled by the second argument. In this example code, 
we set the second argument to 0.1 times the perimeter of the contour, which should provide a reasonably accurate 
approximation for rectangular objects. However, you may need to adjust this value depending on your specific dataset.
"""
# convert the mask image to grayscale
refer_gray = cv2.cvtColor(refer_mask_image, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image to obtain a binary image
refer_, refer_thresh = cv2.threshold(refer_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# find contours in the binary image
refer_contours, refer_ = cv2.findContours(refer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# show
# contours_img = cv2.drawContours(source_image.copy(), refer_contours, -1, (0, 0, 255), 2)
# cv2.imwrite(SAVE_PATH + "\\refer_gray.jpg", refer_gray)
# cv2.imwrite(SAVE_PATH + "\\refer_thresh.jpg", refer_thresh)
# cv2.imwrite(SAVE_PATH + "\\refer_contours.jpg", contours_img)
# cv2.imshow("Gray", refer_gray)
# cv2.imshow("thresh", refer_thresh)
# cv2.imshow("ref_Contours", contours_img)
# cv2.waitKey(0)

# loop through the contours and filter out those that are not rectangular
refer_rects = []
for refer_contour in refer_contours:
    perimeter = cv2.arcLength(refer_contour, True)
    approx = cv2.approxPolyDP(refer_contour, 0.1 * perimeter, True)
    if len(approx) == 4:
        refer_rects.append(approx.reshape(-1, 2))
        # show
        approx_img = cv2.drawContours(source_image.copy(), approx, -1, (0, 255, 0), 2)
        # cv2.imshow("Approx", approx_img)
        # cv2.waitKey(0)

"""
the rectangular mask of the reference object is not a standard rectangle but instead has a tilt angle, 
we need to adjust the method used to calculate the pixel height (ref_height_px).
we use the minimum bounding rectangle (MBR) of the reference object mask to estimate its height. 
The MBR is the smallest rectangle that can fully enclose the object mask, regardless of its orientation or shape
"""

# Loop through each of the rectangular reference objects and calculate their aspect ratio and angle.

# angles = []
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

    # calculate the aspect ratio
    # w = euclidean_distance(sorted_box[0], sorted_box[1])
    # h = euclidean_distance(sorted_box[0], sorted_box[3])

    w = euclidean_distance(p1, p2)
    h = euclidean_distance(p2, p3)
    if w > h:
        temp = h
        h = w
        w = temp
    aspect_ratios.append(w / h)

    # show
    # box_img = cv2.drawContours(source_image.copy(), [refer_rect], -1, (0, 255, 0), 2)
    # cv2.imshow("Box", box_img)
    # cv2.waitKey(0)

    # calculate the angle
    # angle = math.atan2(refer_rect[1][1] - refer_rect[0][1], refer_rect[1][0] - refer_rect[0][0])
    # angles.append(angle)

# Choose the reference object with the closest aspect ratio to the expected aspect ratio of the reference object.
expected_aspect_ratio = ref_width_mm/ref_height_mm
best_rect_idx = np.argmin(np.abs(np.array(aspect_ratios) - expected_aspect_ratio))
best_rect = sorted_boxes[best_rect_idx]

# show
best_img = cv2.drawContours(source_image.copy(), [best_rect], -1, (0, 0, 255), 2)
cv2.imshow("Best_ref", best_img)
# cv2.imwrite(SAVE_PATH + "\\best_ref.jpg", best_img)

# calculate the height of the reference object in the image, calculate the pixel height of the MBR

indexb, p1b, p2b, p3b = find_nearest_90_degree_angle(best_rect)


############################################################################################################################
rect_ref_pixels = np.array([best_rect[0], best_rect[1], best_rect[2], best_rect[3]], dtype=np.float32)  # 矩形参照物的最小外接矩形的四个顶点坐标
correct_ref_image, M = correct_perspective(best_img, rect_ref_pixels, ref_width_mm, ref_height_mm)
cv2.imshow("correct_ref", correct_ref_image)
cv2.waitKey(0)

ref_width_px = ref_width_mm
ref_height_px = ref_height_mm

# ref_width_px = euclidean_distance(p1b, p2b)
# ref_height_px = euclidean_distance(p2b, p3b)
if ref_width_px > ref_height_px:
    ref_temp = ref_height_px
    ref_height_px = ref_width_px
    ref_width_px = ref_temp
############################################################################################################################

"""
find the minimum circumscribed rectangle of the soybean mask in the soybean plant mask picture 
and calculate the pixel height of the minimum circumscribed rectangle
"""

# convert the soybean mask to grayscale
soybean_gray = cv2.cvtColor(soybean_mask_image, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image to obtain a binary image
soybean_, soybean_thresh = cv2.threshold(soybean_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# show
# soybean_contours_t, soybean_t = cv2.findContours(soybean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# soy_contours_t = cv2.drawContours(source_image.copy(), soybean_contours_t, -1, (0, 255, 0), 2)
# cv2.imshow("soy_contours_beforeLinked", soy_contours_t)
# show
# cv2.imshow("sour", soybean_thresh)
# cv2.imwrite(SAVE_PATH + "\\soybean_gray.jpg", soybean_gray)
# cv2.imwrite(SAVE_PATH + "\\soybean_thresh.jpg", soybean_thresh)

# Perform a morphological closing operation to connect the regions
# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(soybean_thresh)

# Loop until there is only one connected component
kernel_size = 4
while num_labels > 2:
    # Perform connected component closure
    kernel_size = kernel_size + 1
    c_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    soybean_thresh = cv2.morphologyEx(soybean_thresh, cv2.MORPH_CLOSE, c_kernel)
    # Find connected components again
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(soybean_thresh)

# find contours in the binary image
soybean_contours, soybean_ = cv2.findContours(soybean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# find the largest contour
soybean_contours_sorted = sorted(soybean_contours, key=cv2.contourArea, reverse=True)[:1]

# find the minimum bounding rectangle of the soybean mask
soybean_rect = cv2.minAreaRect(soybean_contours_sorted[0])

# get the minimum bounding rectangle of the soybean mask
soybean_box = cv2.boxPoints(soybean_rect)
soybean_box = np.int0(soybean_box)

# sort the vertices of the MBR in clockwise order
soybean_sorted_box = sort_box_vertices(soybean_box)

# calculate the pixel height of the minimum bounding rectangle
# soybean_height_px = soybean_sorted_box[3][1] - soybean_sorted_box[0][1]
soybean_width_px = euclidean_distance(soybean_sorted_box[1][1], soybean_sorted_box[0][1])
soybean_height_px = euclidean_distance(soybean_sorted_box[3][1], soybean_sorted_box[0][1])
if soybean_width_px > soybean_height_px:
    soybean_temp = soybean_height_px
    soybean_height_px = soybean_width_px
    soybean_width_px = soybean_temp

###########################################################################################################################

correct_soy_sorted_points, soybean_width_px, soybean_height_px = transform_rect_points(soybean_sorted_box, M)

###########################################################################################################################



# calculate the height of the soybean plant ''soybean_height_px = soybean_rect[3][1] - soybean_rect[0][1]
soybean_height_mm = (ref_height_mm * soybean_height_px) / ref_height_px

# show
# soy_contours = cv2.drawContours(source_image.copy(), soybean_contours, -1, (0, 255, 0), 2)
# cv2.imshow("soy_contours_linked", soy_contours)
# cv2.imwrite(SAVE_PATH + "\\soy_contours.jpg", soy_contours)
# cv2.waitKey(0)
# isbox = cv2.boxPoints(soybean_rect)
# isbox = np.int0(isbox)
# soy_box = cv2.drawContours(source_image.copy(), [isbox], -1, (0, 255, 0), 2)
# cv2.imshow("soy_box", soy_box)
# cv2.imwrite(SAVE_PATH + "\\soy_box.jpg", soy_box)
# cv2.waitKey(0)


# code to display the image with the selected reference object and soybean plant height
print('ref_height_px = ', ref_height_px)
print('soybean_height_px = ', soybean_height_px)
print('soybean_height_mm = ', soybean_height_mm)
