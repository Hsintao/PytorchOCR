#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/5 3:31 下午
# @Author : Xintao
# @File : vis_result.py

import cv2
import os
import numpy as np

result_txt_dir = '/Users/xintao/Desktop/val_size_1024'
idxs = [i.split('_')[-1].split('.')[0] for i in os.listdir(result_txt_dir)]
for idx in idxs:
    with open(result_txt_dir+'/res_img_{}.txt'.format(str(idx)), 'r') as f:
        preds = [i.strip().split(',') for i in f.readlines()]
    img = cv2.imread('/Volumes/Samsung_T5/自然场景文字检测_自然场景文字检测挑战赛初赛数据/测试集/image/img_{}.jpg'.format(str(idx)))
    pts = []
    for i in preds:
        i = list(map(float, i))
        if float(i[-1]) >=0.5:
            pts.append(np.array(i[:-1], np.int).reshape(-1,1,2))
    #     print(pts)
    cv2.polylines(img, pts, True, (0, 255,255),2)
    result_dir = 'val_size_1024_vis'
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(os.path.join(result_dir, f'res_img_{idx}.jpg'), img)
