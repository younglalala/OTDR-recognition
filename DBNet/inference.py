# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 23:51
# @Author  : zonas.wang
# @Email   : zonas.wang@gmail.com
# @File    : inference.py
import math
import sys
import os

sys.path.append(os.getcwd()+'/DBNet')

print(os.getcwd())
import time
import copy
import image_crop_tools

import tensorflow as tf
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


from models.model import DBNet
from config import DBConfig
cfg = DBConfig()
gpu_id = '-1'

def resize_image(image, image_short_side=736):
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    if not contour.size:
        return [], 0
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=500, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue

        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


class OTDRTextDetection():
    def __init__(self, model_path=None,gpu_id = None):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print("DBnet use gpu ID:{}-+-+-+-+-+-+-+-+-+-+-+-".format(gpu_id))
        self.BOX_THRESH = 0.5
        self.mean = np.array([103.939, 116.779, 123.68])
        self.model = DBNet(cfg, model='inference')
        self.model.load_weights(model_path, by_name=True, skip_mismatch=True)


    def get_model_predict(self, image, _=None):

        src_image = image.copy()
        h, w = image.shape[:2]
        image = resize_image(image)
        image = image.astype(np.float32)
        image -= self.mean
        image_input = np.expand_dims(image, axis=0)
        image_input_tensor = tf.convert_to_tensor(image_input)
        start_time = time.time()
        p = self.model.predict(image_input_tensor)[0]
        end_time = time.time()
        print("time: ", end_time - start_time)

        bitmap = p > 0.3
        boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=self.BOX_THRESH)

        all_crop_boxes = []
        all_crop_img = []

        for box in boxes:
            rect = cv2.minAreaRect(np.array(box))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
            bbox = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
            bbox = np.int0(bbox)
            # 坐标点集合
            if np.linalg.norm(bbox[0] - bbox[1]) < 5 or np.linalg.norm(bbox[3] - bbox[0]) < 5:
                continue
            # new_bbox = np.array([[bbox[0, 0], bbox[0, 1]], [bbox[1, 0], bbox[1, 1]], [bbox[2, 0], bbox[2, 1]],
            #                      [bbox[3, 0], bbox[3, 1]]])


            # 图像截取
            box1 = sorted(bbox, key=lambda x: (x[1], x[0]))

            # 将坐标点按照顺时针方向来排序，如果未进行此操作，box的从左往右从上到下排序
            if box1[0][0] > box1[1][0]:
                box1[0], box1[1] = box1[1], box1[0]
            if box1[2][0] < box1[3][0]:
                box1[2], box1[3] = box1[3], box1[2]
            if box1[0][1] > box1[1][1]:
                box1[0], box1[1], box1[2], box1[3] = box1[1], box1[2], box1[3], box1[0]
            tmp_box = copy.deepcopy(np.array(box1)).astype(np.float32)
            if len(tmp_box[tmp_box < 0]) != 0:
                tmp_box[tmp_box < 0] = 0
            # print(tmp_box,'**'*10)
            boxes = tmp_box.copy()

            all_crop_boxes.append(boxes)
            partImg_array = image_crop_tools.get_rotate_crop_image(src_image, tmp_box)
            # x1, y1, x2, y2, x3, y3, x4, y4 = tmp_box[0][0], tmp_box[0][1], tmp_box[1][0], tmp_box[1][1], tmp_box[2][0], \
            #                                  tmp_box[2][1], tmp_box[3][0], tmp_box[3][1]



            all_crop_img.append(partImg_array)

        return all_crop_img,all_crop_boxes



if __name__ == "__main__":


    import sys
    DetectionModelPath = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/models/db_167_1.9499_1.9947.h5'
    detection_model = OTDRTextDetection(model_path=DetectionModelPath,gpu_id = '-1')
    img_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/OTDR2'
    save_path = '/node4_gpu_nfs_raid10/wy/a_shandong/OTDR-recognition-data/crop_out'
    c = 0

    for file in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, file))[:, :, ::-1]
        all_crop_img, all_crop_bbox = detection_model.get_model_predict(img)
        for crop_img in all_crop_img:
            crop_name = "{}.jpg".format(c)
            cv2.imwrite(os.path.join(save_path, crop_name),crop_img)
            c += 1

