import json
import os
import base64
import cv2

# json_save_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR_json'
# txt_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR_txt'   #psenet输出的txt
# img_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR2'
# null = 0
# def txt2json(txt_path,json_save_path,img_path):
#     """
#     :param txt_path: psenet输出的txt
#     :param json_save_path: 生成的json文件的保存路径
#     :param img_path:图片的路径
#     :return:
#     """
#
#
#     for txt_file in os.listdir(txt_path):
#         base_name = txt_file.split('.')[0]
#
#         if txt_file.split('.')[-1] == 'txt':
#             #读取txt里面的坐标信息
#             txt_f = open(os.path.join(txt_path,txt_file),'r',encoding='utf-8')
#             txt_infor = [i.rstrip() for i in txt_f.readlines()]
#
#             #读取图片base64信息
#             img_name =  base_name+'.jpg'
#             # print(os.path.join(img_path, img_name))
#             h, w, c = cv2.imread(os.path.join(img_path, img_name)).shape
#             #获取图片宽高
#
#             img_read = open(os.path.join(img_path,img_name), "rb").read()
#             image_base64 = base64.b64encode(img_read)
#
#
#
#             #构建labelme字典
#             json_dict = {
#                 "version": "4.2.9",
#                 "flags": {},
#                 "shapes": [],
#                 "imagePath": img_name,
#                 "imageData":str(image_base64,'utf8'),
#                 "imageHeight":h,
#                 "imageWidth":w,
#             }
#             for infor in txt_infor:
#                 x1,y1,x2,y2,x3,y3,x4,y4 = infor.split(',')[:8]
#
#                 label = ''.join(infor.split(',')[8:])
#
#                 if len(label)==0:
#                     print(txt_file,'*-'*10)
#                 shape_dict = {
#                     "label": label,
#                     "points":[[float(x1), float(y1)], [float(x2), float(y2)], [float(x3), float(y3)], [float(x4), float(y4)]],
#                     "group_id": null,
#                     "shape_type": "polygon",
#                     "flags": {}
#                 }
#                 json_dict["shapes"].append(shape_dict)
#
#             #生成json文件
#             json_path = os.path.join(json_save_path,base_name+'.json')
#
#             json_file = open(json_path,'w')
#             json.dump(json_dict,json_file, indent=2)
#
#
#
#
# if __name__ == '__main__':
#     txt2json(txt_path,json_save_path,img_path)
import json
import os
import base64

json_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/pse_json'  # 这是json文件存放的位置
txt_file_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/pse_txt'  # 这是保存txt文件的位置

for file in os.listdir(json_path):
    txt_filename = file.split('.')[0]+'.txt'
    txt_file = open(os.path.join(txt_file_path,txt_filename), 'w')

    with open(os.path.join(json_path,file)) as f:
        pop_data = json.load(f)
        # print(pop_data['shapes'])

        for infor in pop_data['shapes']:
            points = infor["points"]
            x1,y1,x2,y2,x3,y3,x4,y4 = points[0][0],points[0][1],points[1][0],points[1][1],\
                                      points[2][0],points[2][1],points[3][0],points[3][1]
            txt_file.writelines("{},{},{},{},{},{},{},{},text\n".format(int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)))






