import os

from PIL import Image
import numpy as np
import re
import cv2
from DBNet.inference import OTDRTextDetection

from recognition_crnn import predict

DetectionModelPath = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/models/db_167_1.9499_1.9947.h5'
ORDR_text_detection = OTDRTextDetection(model_path=DetectionModelPath,gpu_id='-1')


def OCR_rules(all_bboxes_list,all_res_list,w,h):
    """
    根据识别结果提取相应信息
    1、获取整张图片的2/h，2/3w的区域，即左下角区域；
    2、在左下角区域进行正则匹配找到“结束”区域的位置，匹配“结束，结，束，吉”，并且字符中文长度小于4的作为匹配成功；
    3、结束区域找到向右横向的所有框，并且确认框内全为数字内容；
    4、找到所有数字内容了根据一下标注进行提取：
    位置：0.0000
    反射-00.0
    衰减：0.000
    累计损耗：0.00
    插损：无


    :param all_bboxes_list: 文字片段坐标点信息
    :param all_res_list: 文字片段识别结果
    :param w: 整张图片的宽
    :param h: 整张图片的高
    :return: 位置（location）、反射（reflection）、衰减（attenuation）、累计损耗（stack_loss）、插损的值（insertion_loss)）。

    """





if __name__ =='__main__':
    img_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR2'
    save_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/crop_out'
    c = 0
    for file in os.listdir(img_path):
        f = open(os.path.join(save_path,file.split('.')[0]+'.txt'),'w',encoding='utf-8')
        img = cv2.imread(os.path.join(img_path, file))[:, :, ::-1]
        src_image = img.copy()
        all_crop_img, all_crop_bbox = ORDR_text_detection.get_model_predict(img)

        h, w, _ = img.shape

        all_res_out = []

        for ind, crop_im in enumerate(all_crop_img):
            image = Image.fromarray(crop_im)
            crnn_preds, crnn_pred_prob = predict.predict_prob(image, lan='chinese', useStr=None)
            bbox = all_crop_bbox[ind]
            all_res_out.append(crnn_preds)



            # xmin, ymin, xmax, ymax = map(int, [min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1])])


            # cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            #
            # x1,y1,x2,y2,x3,y3,x4,y4 = bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1],bbox[3][0],bbox[3][1]
