#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/4 7:50 下午
# @Author : Xintao
# @File : ensemble.py
import cv2
import numpy as np
import os
from shapely.geometry import *
import math
from tqdm import tqdm


def py_cpu_pnms(bboxs, scores, thresh):
    # 获取检测坐标点及对应的得分

    pts = bboxs

    areas = np.zeros(scores.shape)
    # 得分降序
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

    for il in range(len(pts)):
        # 当前点集组成多边形，并计算该多边形的面积
        a = pts[il]
        poly = Polygon(pts[il])
        areas[il] = poly.area

        # 对剩余的进行遍历
        for jl in range(il, len(pts)):
            polyj = Polygon(pts[jl])
            # 计算两个多边形的交集，并计算对应的面积
            inS = poly.intersection(polyj)
            inter_areas[il][jl] = inS.area
            inter_areas[jl][il] = inS.area

    # 下面做法和nms一样
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = inter_areas[i][order[1:]] / (areas[i] + areas[order[1:]] - inter_areas[i][order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def rotation_point(img, angle=90, point=None):
    cols = img.shape[1]  # w
    rows = img.shape[0]  # h
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    heightNew = int(
        cols *
        math.fabs(
            math.sin(
                math.radians(angle))) +
        rows *
        math.fabs(
            math.cos(
                math.radians(angle))))
    widthNew = int(
        rows *
        math.fabs(
            math.sin(
                math.radians(angle))) +
        cols *
        math.fabs(
            math.cos(
                math.radians(angle))))
    M[0, 2] += (widthNew - cols) / 2
    M[1, 2] += (heightNew - rows) / 2
    img = cv2.warpAffine(img, M, (widthNew, heightNew))
    a = M[:, :2]  # a.shape (2,2)
    b = M[:, 2:]  # b.shape(2,1)
    b = np.reshape(b, newshape=(1, 2))
    a = np.transpose(a)
    len_1 = len(point)
    # point = np.reshape(point, newshape=(len(point) * 4, 2))
    point = np.reshape(point, newshape=(len_1, 2))
    point = np.dot(point, a) + b
    point = np.reshape(point, newshape=(len_1, 2))
    return img, point


img_dir = '/Volumes/WDSSD/文本检测/自然场景文字检测挑战赛初赛数据/测试集/image'
img1_dir = '/Volumes/WDSSD/文本检测/自然场景文字检测挑战赛初赛数据/测试集/image1'
img2_dir = '/Volumes/WDSSD/文本检测/自然场景文字检测挑战赛初赛数据/测试集/image2'
img_names = os.listdir(img_dir)

img_result_dir = '/Users/xintao/Desktop/3/test_results'
img1_result_dir = '/Users/xintao/Desktop/3/test_results1/'
img2_result_dir = '/Users/xintao/Desktop/3/test_results2/'
final_result_dir = '/Users/xintao/Desktop/3/final_result/'
os.makedirs(final_result_dir, exist_ok=True)

for name in tqdm(img_names):
    # print(name)
    idx = name.split('.')[0].split('_')[-1]
    id = idx
    img_path = os.path.join(img_dir, name)
    img_result_path = os.path.join(img_result_dir, f'res_img_{idx}.txt')
    img1_path = os.path.join(img1_dir, name)
    img1_result_path = os.path.join(img1_result_dir, f'res_img_{idx}.txt')
    img2_path = os.path.join(img2_dir, name)
    img2_result_path = os.path.join(img2_result_dir, f'res_img_{idx}.txt')

    img = cv2.imread(img_path)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    with open(img_result_path, 'r', encoding='utf-8') as f:
        img_result = [i.strip().split(',') for i in f.readlines()]
        img_result = [list(map(float, i)) for i in img_result]
    with open(img1_result_path, 'r', encoding='utf-8') as f:
        img1_result = [i.strip().split(',') for i in f.readlines()]
        img1_result = [list(map(float, i)) for i in img1_result]
    with open(img2_result_path, 'r', encoding='utf-8') as f:
        img2_result = [i.strip().split(',') for i in f.readlines()]
        img2_result = [list(map(float, i)) for i in img2_result]

    for idx, r in enumerate(img_result):
        points = np.array(r[:-1]).reshape(-1, 2).astype(np.int)
        conf = r[-1]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        points = np.int0(box)
        conf = r[-1]
        points = points.reshape(-1).tolist()
        points.append(conf)
        img_result[idx] = points

    for idx, r in enumerate(img1_result):
        points = np.array(r[:-1]).reshape(-1, 2).astype(np.int)
        conf = r[-1]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        points = np.int0(box)
        conf = r[-1]
        # points = np.clip(points, a_min=0, a_max=100000)
        _, points = rotation_point(img1, -90, points)
        # print(points)
        points = points.reshape(-1).tolist()
        points.append(conf)
        img1_result[idx] = points

    for idx, r in enumerate(img2_result):
        points = np.array(r[:-1]).reshape(-1, 2).astype(np.int)
        conf = r[-1]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        points = np.int0(box)
        # points = np.clip(points, a_min=0, a_max=100000)
        conf = r[-1]
        _, points = rotation_point(img2, 90, points)
        points = points.reshape(-1).tolist()
        points.append(conf)
        img2_result[idx] = points

    all_result = img_result + img1_result + img2_result
    bboxes = []
    confs = []
    for i in all_result:
        bboxes.append(np.array(i[:-1]).reshape(-1, 2))
        confs.append(i[-1])
    bboxes = np.array(bboxes)
    confs = np.array(confs)
    keep = py_cpu_pnms(bboxes, confs, thresh=0.1)
    box_list = [bboxes[i] for i in keep]
    score_list = [confs[i] for i in keep]

    with open(os.path.join(final_result_dir, f'res_img_{id}.txt'), 'w', encoding='utf-8') as f:
        for box, score in zip(box_list, score_list):
            f.write(','.join(list(map(str, box.reshape(-1).tolist()))))
            f.write(f',{score}\n')