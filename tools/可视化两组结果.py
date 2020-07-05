#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/5 3:36 下午
# @Author : Xintao
# @File : 可视化两组结果.py
import cv2
import os
img1_dir = '/Users/xintao/Desktop/val_size_1024_vis'
img2_dir = '/Users/xintao/Desktop/2/results'
img_names = os.listdir(img1_dir)
for name in img_names:
    im1 = cv2.imread(os.path.join(img1_dir, name))
    im2 = cv2.imread(os.path.join(img2_dir, name))
#     print(im1.shape, im2.shape)
    im = np.hstack((im1, im2))
#     print(im.shape)
    cv2.imwrite(os.path.join('/Users/xintao/Desktop/ensemble_result/concat', name), im)