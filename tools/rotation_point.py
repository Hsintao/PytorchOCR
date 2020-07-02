#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/2 5:11 下午
# @Author : Xintao
# @File : rotation_point.py
import math

import cv2
import numpy as np


def rotation_point(img, angle=90, point=None):
    cols = img.shape[1]  # w
    rows = img.shape[0]  # h
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    heightNew = int(cols * math.fabs(math.sin(math.radians(angle))) + rows * math.fabs(math.cos(math.radians(angle))))
    widthNew = int(rows * math.fabs(math.sin(math.radians(angle))) + cols * math.fabs(math.cos(math.radians(angle))))
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


img = cv2.imread('../imgs/img_5229.jpg')
point = np.array([[100, 201], [550, 201], [235, 251],
                  [550, 500], [100, 500]], np.int32)
cv2.polylines(img, [point], True, (0, 255, 255))

point_tmp = point.copy()
img_rot, point_rt = rotation_point(img, 90, point_tmp)
# point_rt.astype(np.int32)
point_rt_rt = point_rt.astype(int)
cv2.polylines(img_rot, [point_rt_rt], True, (255, 0, 255), 5)
cv2.imshow('img_rot', img_rot)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
