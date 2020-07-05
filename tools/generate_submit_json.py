#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/5 3:30 下午
# @Author : Xintao
# @File : generate_submit_json.py
# 生成提交的json
import json
import os

result_txt_dir = '/Users/xintao/Desktop/test_results1152/'
idxs = [int(i.split('_')[-1].split('.')[0]) for i in os.listdir(result_txt_dir)]
idxs.sort()
result_json = {}
for idx in idxs:
    l = list()
    with open(result_txt_dir + '/res_img_{}.txt'.format(str(idx)), 'r') as f:
        preds = [i.strip().split(',') for i in f.readlines()]
    for pred in preds:
        points = pred[:-1]
        points = list(map(float, points))
        p = [[int(points[i * 2]), int(points[i * 2 + 1])] for i in range(int(len(points) // 2))]
        points = [p[0]] + p[1:][::-1]
        confidence = float(pred[-1])
        if confidence >= 0.5:
            l.append({"points": points, "confidence": confidence})
    result_json['res_' + str(idx)] = l
with open('db_epoch1100_size1152.json', 'w', encoding='utf-8') as f:
    json.dump(result_json, f, indent=4)
