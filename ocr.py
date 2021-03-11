import os

from PIL import Image
import numpy as np
import re
# # 指定显卡
from recognition_deeptext.recognition_predict import DeeptextRecognition
deeptext = DeeptextRecognition()
from detection.detection_predict import TextDetection

textTFDet = TextDetection()
from recognition_crnn import predict



import time
import cv2



def OCR(img, detection_model, deeptext_model,crnn_model):
    """
    :param img: nparray 图片
    :param detection_moel: 检测部分
    :param recognition_model: 识别部分
    :return:
    """

    crop_img, bbox = detection_model.get_model_predict(img)

    h,w,_ = img.shape
    
    all_out = []
    all_out_crnn = []
    for ind,crop_im in enumerate(crop_img):
        
        deep_preds,deep_pred_prob,confidence_score = deeptext_model.get_model_predict(crop_im)
        image = Image.fromarray(crop_im)
        crnn_preds,crnn_pred_prob = crnn_model.predict_prob(image, lan='chinese', useStr=None)
        print(deep_preds,confidence_score)
        #字符串置信度小于0.5的综合crnn和deeptext识别结果
#         ret = "".join(re.findall("[0-9]", deep_preds))
        if confidence_score == []:
            confidence_score = 0
        if confidence_score <0.2:
            
            crnn_preds_list = list(crnn_preds)
            deep_preds_list = list(deep_preds)
            if len(crnn_preds_list) == len(deep_preds_list):
                for ind, d_prob in enumerate(deep_pred_prob):
                    if crnn_pred_prob[ind] > d_prob:
                        deep_preds_list[ind] = crnn_preds_list[ind]
        

                    

            deep_preds = ''.join(deep_preds_list)
            
        all_out.append(deep_preds)
        all_out_crnn.append(crnn_preds)
    return all_out,bbox,h,w,all_out_crnn

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False

def chinese_count(string):
    hans_total = 0
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fef':
            hans_total += 1
    return hans_total

def ocr_rules(bbox,all_res,all_out_crnn):
    """
    1、非长方形框，宽//高在1.5倍以上则保留；
    2、以最长边为基准，计算各个框的面积，面积1/10<最长边面积<1.5
    :param bbox:ocr输出的所有bbox
    :param all_res:ocr输出的所有文字信息结果
    :return:经过赛选的ocr bbox和文字识别结果
    """
    setoff_list = []
    area_list = []
    for box in bbox:
        x_list, y_list = box[:, 0], box[:, 1]
        x1, y1, x2, y2, x3, y3, x4, y4 = x_list[0], y_list[0], x_list[1], y_list[1], x_list[2], y_list[2], x_list[3], \
                                         y_list[3]
        box_h, box_w = y4 - y1, x2 - x1
        
        xmin,ymin,xmax,ymax =map(int,[min(box[:,0]),min(box[:,1]),max(box[:,0]),max(box[:,1])])
        setoff_list.append((xmax - xmin) / (ymax - ymin))
        area_list.append((ymax - ymin)*(xmax - xmin))

    if len(setoff_list) > 0:
        max_setoff_index = setoff_list.index(max(setoff_list))
        max_area = max(area_list)

        last_bbox = []
        last_res = []

        for ind,infor in enumerate(setoff_list):
            #2、以最长边为基准，计算各个框的面积，面积1/10<最长边面积<1.5
            if infor >= 1.5 and 0.1< area_list[ind]/max_area <1.5:
#                 last_bbox.append(bbox[ind])
#                 last_res.append(all_res[ind])
                ch_count = chinese_count(all_out_crnn[ind])
                #3、字符串中中文的个数占整个字符串的0.3或者是2个中文以上的进行过滤，
                if len(all_out_crnn[ind]) !=0 and ch_count/len(all_out_crnn[ind]) >= 0.3 and all_out_crnn[ind] != " " :
                    pass
                ret = "".join(re.findall("[0-9]", all_out_crnn[ind]))
                if len(ret) >= 8 and ret.isdigit():
                    pass
                
                last_bbox.append(bbox[ind])
                last_res.append(all_res[ind])

        return last_res,last_bbox
    else:
        return all_res,bbox





if __name__ == "__main__":
    pass
    
    
    #####deeptext test#########
#     img_path = '/ntt/scm_yzy2_model/deep-text-recognition/test_new'
#     f = open('result.txt','w',encoding = 'utf-8')
#     crop_img = []
#     all_file = []
#     for file in os.listdir(img_path):

#         img = np.array(Image.open(os.path.join(img_path,file)))
#         crop_img.append(img)
#         all_file.append(file)
    
#     for ind,crop_im in enumerate(crop_img):
#         try:
#             deep_preds,deep_pred_prob,confidence_score = deeptext.get_model_predict(crop_im)
#             #字符串置信度小于0.5的综合crnn和deeptext识别结果
#             ret = "".join(re.findall("[0-9]", deep_preds))
#             if confidence_score <0.4 and len(ret)/len(deep_pred_prob)<0.3:
#                 image = Image.fromarray(crop_im)
#                 crnn_preds,crnn_pred_prob = predict.predict_prob(image, lan='chinese', useStr=None)
#                 crnn_preds_list = list(crnn_preds)
#                 deep_preds_list = list(deep_preds)
#                 if len(crnn_preds_list) == len(deep_preds_list):
#                     for ind, d_prob in enumerate(deep_pred_prob):
#                         if crnn_pred_prob[ind] > d_prob:
#                             deep_preds_list[ind] = crnn_preds_list[ind]




#                 deep_preds = ''.join(deep_preds_list)    
#             f.writelines("{} {}\n".format(all_file[ind],deep_preds))
#             print("{} {}\n".format(all_file[ind],deep_preds))
#         except:
#             pass
  
    
    #################ocr test######################
#     img_path = '/ntt/wy_home/gjx_test'
#     qr_result = '光交箱喷码识别.txt'
#     f = open(qr_result, 'w', encoding='utf-8')
#     c = 0

#     for ordername in os.listdir(img_path):
#         orderinfor = []
#         order_path = os.path.join(img_path, ordername)
#         # print(order_path)
#         for img_file in os.listdir(order_path):
#             if img_file != '.ipynb_checkpoints':
#                 img_path2 = os.path.join(order_path, img_file)
#                 print(img_path2)
#                 img = cv2.imread(img_path2)[:, :, ::-1]


#                 res = OCR(img, textTFDet, deeptext, predict)
#                 last_res = ocr_rules(res[1], res[0], res[-1])
#                 if len(''.join(res[0]))>10:
#                     orderinfor.append(1)
#                 copy_img = np.copy(img)
#                 for indx, box in enumerate(last_res[1]):
#                     xmin, ymin, xmax, ymax = map(int, [min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])])
#                     #             print(box)

#                     #             print(xmin,ymin,xmax,ymax)

#                     cv2.rectangle(copy_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

#                     cv2.putText(copy_img, last_res[0][indx], (0, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
#                     # cv2.imwrite(os.path.join("res_out", file), copy_img[:, :, ::-1])
#                 #         f.writelines('{} {}\n'.format(file, res[0]))
#         if 1 in orderinfor:
#             f_res = 1
#         else:
#             f_res = 0
#         f.writelines('{} {}\n'.format(ordername, f_res))
#         print('{} {}\n'.format(ordername, f_res))
#     f.close()
#####################################################
    img_path = '/ntt/OCR-new-data/sc_dp_test1'
#     txt_path = '/ntt/scm_yzy2_model/tensorflow_PSENet_train_data/dxdp_test_txt2'
    res_out_path = '/ntt/OCR-new-data/res_out'

    c = 0


    for img_file in os.listdir(img_path):

#         txt_file = img_file.split('.')[0]+'.txt'
#         f = open(os.path.join(txt_path,txt_file), 'w', encoding='utf-8')

        img_path2 = os.path.join(img_path, img_file)
        img = cv2.imread(img_path2)[:, :, ::-1]
        #ocr 识别结果
        res = OCR(img, textTFDet, deeptext, predict)
        #输出结果后处理（基于规则筛选输出最终结果）
        last_res = ocr_rules(res[1], res[0], res[-1])
       
        copy_img = np.copy(img)
        for indx1, box in enumerate(res[1]):
            xmin, ymin, xmax, ymax = map(int, [min(box[:, 0]), min(box[:, 1]), max(box[:, 0]), max(box[:, 1])])

            
            cv2.rectangle(copy_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            try:
                cv2.putText(copy_img, last_res[0][indx1], (0, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.imwrite(os.path.join(res_out_path, img_file), copy_img[:, :, ::-1])
            except:
                pass
            
#             print(box,'*-'*10)
#             f.writelines('{},{},{},{},{},{},{},{},{}\n'.format(box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],res[0][indx]))




#     img_path = '/ntt/scm_yzy2_model/tensorflow_PSENet_train_data/dxdp_test2'
#     txt_path = '/ntt/scm_yzy2_model/tensorflow_PSENet_train_data/dxdp_test_crop2'
    
#     c = 0
         
#     for img_file in os.listdir(img_path):

#         print(img_file)
#         img_path2 = os.path.join(img_path, img_file)
#         img = cv2.imread(img_path2)[:, :, ::-1]    
#         crop_img, bbox = textTFDet.get_model_predict(img)   
#         for c_img in crop_img:
#             cv2.imwrite(os.path.join(txt_path,"crop_{}.jpg".format(c)),c_img[:,:,::-1])
# #             cv2.imwrite("2.jpg",c_img[:,:,::-1])
    
#             c+=1
        
        
        