import os

from PIL import Image
import numpy as np
import re
import cv2
from DBNet.inference import OTDRTextDetection

from recognition_crnn import predict

DetectionModelPath = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/models/db_167_1.9499_1.9947.h5'
ORDR_text_detection = OTDRTextDetection(model_path=DetectionModelPath,gpu_id='-1')






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

        all_out = []
        all_out_crnn = []
        print('*-'*10,file,'*-'*10)
        for ind, crop_im in enumerate(all_crop_img):
            image = Image.fromarray(crop_im)
            crnn_preds, crnn_pred_prob = predict.predict_prob(image, lan='chinese', useStr=None)
            print(crnn_preds)
            bbox = all_crop_bbox[ind]
            print(bbox)
            xmin, ymin, xmax, ymax = map(int, [min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1])])


            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(save_path,file),src_image)
            x1,y1,x2,y2,x3,y3,x4,y4 = bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1],bbox[3][0],bbox[3][1]
            f.writelines('{},{},{},{},{},{},{},{},{}\n'.format(x1,y1,x2,y2,x3,y3,x4,y4,crnn_preds))
        print('*-' * 10, file, '*-' * 10)