from PIL import Image
import numpy as np
import os
n=2227 #n为.json文件个数
LABELME_JSON_DIR = os.path.join("D:\\All_Codes\\MaskRCNNTest\\Mask_RCNN\\samples\\plant2227_dataset\\labelme_json")
count = os.listdir(LABELME_JSON_DIR)
for i in range(n):
    # open_path='E:/data_image/new_file/L_train/labelme_json/'+'L'+format(str(i), '0>1s')+'_json'+'/label.png'#文件地址
    # open_path = "D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\plant100_badDataset\labelme_json/Soybean"+format(str(i+1), '0>4s')+'_json'+'/label.png'
    open_path = "D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\\plant2227_dataset\labelme_json/%s" % str(
        count[i][0:11]) + '_json' + '/label.png'
    # try:
    #     f=open(open_path)
    #     f.close()
    # except FileNotFoundError:
    #     continue
    img1=Image.open(open_path)#打开图像
    print(img1)
    save_path='D:\All_Codes\MaskRCNNTest\Mask_RCNN\samples\\plant2227_dataset\cv2_mask/'#保存地址
    # img1.show()
    # img=Image.fromarray(np.uint8(img1))#16位转换成8位
    img=img1
    # img.save(os.path.join(save_path,'Soybean'+format(str(i+1),'0>4s')+'.png')) #保存成png格式
    img.save(os.path.join(save_path, '%s' % str(count[i][0:11]) + '.png'))  # 保存成png格式