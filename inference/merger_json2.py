# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:31:49 2018

@author: wangq
"""

import numpy as np
import os
import json
from time import time

def get_all_img_in_json(json_path):

    with open(json_path, 'r') as load_f:
        json_data = json.load(load_f)

    all_images = []
    for i, box in enumerate(json_data):
        name = box['name']
        if name not in all_images:
            all_images.append(name)
    return all_images


def nms(dets, thresh=0.3):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def merger_json(): 

    json_map_path = '../data/result_map.json'  
    json_zj_path = '../data/result_zj.json'
    json_output = '../submit/result' + str(time.time()) + '.json'  # 输出json
    th_nms = 0.5
    vote = False
    th_vote = 0.5

    result = []

    with open(json_map_path, 'r') as load_f:
        json_data_1 = json.load(load_f)
    with open(json_acc_path, 'r') as load_f:
        json_data_2 = json.load(load_f)
    with open(json_zj_path, 'r') as load_f:
        json_data_3 = json.load(load_f)

    img_list = get_all_img_in_json(json_acc_path)

    for name in img_list:
        boxes = []

        for box in json_data_1:
            img_name = box['name']
            if name != img_name:
                continue
            else:
                category = box['category']
                bbox = box['bbox']
                score = box['score']
                boxes.append(bbox + [score] + [category])
        for box in json_data_2:
            img_name = box['name']
            if name != img_name:
                continue
            else:
                category = box['category']
                bbox = box['bbox']
                score = box['score']
                boxes.append(bbox + [score] + [category])
        for box in json_data_3:
            img_name = box['name']
            if name != img_name:
                continue
            else:
                category = box['category']
                bbox = box['bbox']
                score = box['score']
                boxes.append(bbox + [score] + [category])

        boxes = np.array(boxes)
        for kind in range(1, 15+1, 1):
            boxes_kind = boxes[boxes[:, -1] == kind][:, :-1]
            order_nms = nms(boxes_kind, thresh=th_nms)
            boxes_kind = boxes_kind[order_nms]

            if vote:
                boxes_kind_raw = boxes[boxes[:, -1] == kind][:, :-1]
                boxes_kind = box_voting(boxes_kind, boxes_kind_raw, th_vote)

            if len(boxes_kind) <= 0:
                continue
            for box in boxes_kind:
                l = {'name': name,
                     'category': kind,
                     'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                     'score': float(box[4])}
                # print(l)
                result.append(l)

    with open(json_output, 'w') as fp:
         json.dump(result, fp, indent=4, separators=(',', ': '))


def box_iou_vote(b1, b2):

    b1 = np.expand_dims(b1, -2)
    b2 = np.expand_dims(b2, 0)

    b1_mins = b1[..., :2]
    b1_maxes = b1[..., 2:4]
    b1_wh = b1[..., 2:4] - b1[..., 0:2]

    b2_mins = b2[..., :2]
    b2_maxes = b2[..., 2:4]
    b2_wh = b1[..., 2:4] - b1[..., 0:2]

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_voting(boxes, boxes_all, th_vote):
    mask_iou = box_iou_vote(boxes, boxes_all)
    mask_iou = mask_iou >= th_vote

    for i, box in enumerate(boxes):
        boxes_sample = boxes_all[mask_iou[i]]
        boxes_sample = boxes_sample[:, :4] * boxes_sample[:, -1:] / np.sum(boxes_sample[:, -1:])
        boxes[i, :4] = np.sum(boxes_sample[:, :4], axis=0)

    return boxes


if __name__ == '__main__':
    merger_json()







