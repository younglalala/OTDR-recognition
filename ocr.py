import os

from PIL import Image
import numpy as np
import re
import cv2
from DBNet.inference import OTDRTextDetection

from recognition_crnn import predict

DetectionModelPath = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/models/db_167_1.9499_1.9947.h5'
ORDR_text_detection = OTDRTextDetection(model_path=DetectionModelPath,gpu_id='-1')
def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
def containenglish(str0):
    import re
    return bool(re.search('[A-Za-z]', str0))
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def OCR_rules(all_bboxes_list,all_res_list,w,h):
    """
    根据识别结果提取相应信息
    1、获取整张图片的2/h，2/3w的区域，即左下角区域；
    2、在左下角区域进行正则匹配找到“结束”区域的位置，匹配“结束，结，束，吉”，并且字符中文长度小于4的作为匹配成功；
    3、结束区域找到向右横向的所有框，并且确认框内全为数字内容；
    4、找到所有数字内容了根据一下标注进行提取：
    位置location：0.0000
    反射reflection-00.0
    衰减attenuation：0.000
    累计损耗stack_loss：0.00
    插损insertion_loss：无


    :param all_bboxes_list: 文字片段坐标点信息
    :param all_res_list: 文字片段识别结果
    :param w: 整张图片的宽
    :param h: 整张图片的高
    :return: 位置（location）、反射（reflection）、衰减（attenuation）、累计损耗（stack_loss）、插损的值（insertion_loss)）。

    """

    #1、获取整张图片的2/h，2/3w的区域，即左下角区域；
    stage1_res = []
    stage1_centerroi = []
    for indx,bboxes in enumerate(all_bboxes_list):

        bboxes_miny = min(bboxes[:,1])
        bboxes_maxy = max(bboxes[:,1])
        bboxes_minx = min(bboxes[:,0])
        bboxes_maxx = max(bboxes[:, 0])
        if bboxes_miny >= h/2 and bboxes_minx <= 2*w/3:

            centerroi= [(bboxes_maxx-bboxes_minx) / 2 +bboxes_minx,(bboxes_maxy-bboxes_miny) / 2 +bboxes_miny]
            stage1_centerroi.append(centerroi)
            stage1_res.append(all_res_list[indx])
    stage1_dict = dict(zip(stage1_res,stage1_centerroi))

    # 2、在左下角区域进行正则匹配找到“结束”区域的位置，匹配“结束，结，束，吉”，并且字符中文长度小于4的作为匹配成功；
    stage2_res = []
    stage2_centerroi = []
    cut_infor = []
    for indx2,stage1_infor in enumerate(stage1_res) :
        if '结束' in stage1_infor or '结' in stage1_infor or '束' in stage1_infor or '吉' in stage1_infor :
            if len(stage1_infor) <5 :
                cut_infor.append(stage1_infor)
                stage2_res.append(stage1_infor)
                stage2_centerroi.append(stage1_centerroi[indx2])

        #3、判断是否有中文，有中文的不保留
        if not check_contain_chinese(stage1_infor) and not containenglish(stage1_infor) and '.' in stage1_infor:
            stage2_res.append(stage1_infor)
            stage2_centerroi.append(stage1_centerroi[indx2])
    stage2_dict = dict(zip(stage2_res, stage2_centerroi))
    print(stage2_dict)
    # 位置location：0.0000
    # 反射reflection - 00.0
    # 衰减attenuation：0.000
    # 累计损耗stack_loss：0.00
    # 插损insertion_loss：无

    all_res_dict = {
        "location":None,
        "reflection":None,
        "attenuation":None,
        "stack_loss":None,
        "insertion_loss":''
    }


    if len(cut_infor) > 0:
        flag_infor_y = stage2_dict[cut_infor[0]][-1]


        #反射信息
        reflectionres_list = []
        reflectiony_list = []


        #位置信息
        locationres_list = []
        locationy_list = []

        #衰减信息
        attenuationres_list = []
        attenuationy_list = []


        #累计损失信息
        stack_lossres_list = []
        stack_lossy_list = []

        for infor1 in stage2_dict:
            #获取反射内容

            reflection_res = ''.join(re.findall('-[0-9]{2}.[0-9]{1}', infor1))
            if reflection_res != '':
                reflectionres_list.append(reflection_res)
                reflectiony_list.append(stage2_dict[infor1][-1])

            #获取位置信息
            if len(infor1) == 6 and infor1.count('.') ==1:
                locationres_list.append(infor1)
                locationy_list.append(stage2_dict[infor1][-1])
            if len(infor1) == 11 and infor1.count('.') ==2 and infor1.count('-') ==1:
                locationres_list.append(infor1.split("-")[0])
                locationy_list.append(stage2_dict[infor1][-1])

            # 获取衰减信息
            if len(infor1) == 5 and infor1.count('.') == 1 and '-' not in infor1:
                attenuationres_list.append(infor1)
                attenuationy_list.append(stage2_dict[infor1][-1])
            if len(infor1) == 9 and infor1.count('.') == 2 and  '-' not in infor1:
                attenuationres_list.append(infor1[:5])
                attenuationy_list.append(stage2_dict[infor1][-1])

                stack_lossres_list.append(infor1[5:])
                stack_lossy_list.append(stage2_dict[infor1][-1])


            # 获取累计损失信息
            if len(infor1) == 4 and infor1.count('.') == 1 and '-' not in infor1:
                stack_lossres_list.append(infor1)
                stack_lossy_list.append(stage2_dict[infor1][-1])




        #反射   后处理部分
        last_reflection_ylist = np.array(reflectiony_list) - int(flag_infor_y)
        abs_last_reflection_ylist = np.maximum(last_reflection_ylist, -last_reflection_ylist)
        if len(abs_last_reflection_ylist) >0:
            last_reflection_index = reflectionres_list[np.argmin(abs_last_reflection_ylist)]    #求绝对值
            all_res_dict["reflection"] = last_reflection_index

        # 位置   后处理部分
        last_location_ylist = np.array(locationy_list) - int(flag_infor_y)
        abs_last_location_ylist = np.maximum(last_location_ylist, -last_location_ylist)
        if len(abs_last_location_ylist) > 0:
            last_location_index = locationres_list[np.argmin(abs_last_location_ylist)]  # 求绝对值
            all_res_dict["location"] = last_location_index

        # 衰减信息   后处理部分
        last_attenuation_ylist = np.array(attenuationy_list) - int(flag_infor_y)
        abs_last_attenuation_ylist = np.maximum(last_attenuation_ylist, -last_attenuation_ylist)
        if len(abs_last_attenuation_ylist) > 0:
            last_attenuation_index = attenuationres_list[np.argmin(abs_last_attenuation_ylist)]  # 求绝对值
            all_res_dict["attenuation"] = last_attenuation_index



        # 累计损失   后处理部分
        last_stack_loss_ylist = np.array(stack_lossy_list) - int(flag_infor_y)
        abs_stack_loss_ylist = np.maximum(last_stack_loss_ylist, -last_stack_loss_ylist)
        if len(abs_stack_loss_ylist) > 0:
            last_stack_loss_index = stack_lossres_list[np.argmin(abs_stack_loss_ylist)]  # 求绝对值
            all_res_dict["stack_loss"] = last_stack_loss_index


    return all_res_dict









if __name__ =='__main__':
    img_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR2'
    save_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/crop_out'
    c = 0
    res_f = open('stage1_res.txt','w',encoding="utf-8")
    for file in os.listdir(img_path):

        # f = open(os.path.join(save_path,file.split('.')[0]+'.txt'),'w',encoding='utf-8')

        img = cv2.imread(os.path.join(img_path, file))[:, :, ::-1]
        print(file)

        src_image = img.copy()
        all_crop_img, all_crop_bbox = ORDR_text_detection.get_model_predict(img)

        h, w, _ = img.shape

        all_res_out = []

        for ind, crop_im in enumerate(all_crop_img):
            image = Image.fromarray(crop_im)
            crnn_preds, crnn_pred_prob = predict.predict_prob(image, lan='chinese', useStr=None)
            bbox = all_crop_bbox[ind]
            all_res_out.append(crnn_preds)

        res_dict = OCR_rules(all_crop_bbox,all_res_out,w,h)
        print(res_dict)
        res_f.writelines("{} {}\n".format(file,res_dict))
    res_f.close()

            # xmin, ymin, xmax, ymax = map(int, [min(bbox[:, 0]), min(bbox[:, 1]), max(bbox[:, 0]), max(bbox[:, 1])])


            # cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            #
            # x1,y1,x2,y2,x3,y3,x4,y4 = bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1],bbox[3][0],bbox[3][1]
