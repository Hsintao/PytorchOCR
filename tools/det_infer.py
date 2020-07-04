# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib
# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from shapely.geometry import *
import numpy as np
import shutil
from tqdm import tqdm
from torchocr.utils import draw_ocr_box_txt, draw_bbox
import math
import cv2
import time
import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize
from torchocr.postprocess import build_post_process


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.resize = ResizeShortSize(736, False)
        self.post_proess = build_post_process(cfg['post_process'])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        with torch.no_grad():
            tensor = tensor.to(self.device)
            out = self.model(tensor)
        box_list, score_list = self.post_proess(
            out, data['shape'], is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument(
        '--model_path',
        required=True,
        type=str,
        help='rec model path')
    parser.add_argument(
        '--img_path',
        required=True,
        type=str,
        help='img path for predict')
    args = parser.parse_args()
    return args


def rotation_point(img, angle=90, point=None):
    print(point.shape)
    # point.shape  (N,2)
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


# coding=utf-8


def py_cpu_pnms(bboxs, scores, thresh):
    # 获取检测坐标点及对应的得分

    pts = bboxs

    areas = np.zeros(scores.shape)
    # 得分降序
    order = scores.argsort()[::-1]
    inter_areas = np.zeros((scores.shape[0], scores.shape[0]))

    for il in range(len(pts)):
        # 当前点集组成多边形，并计算该多边形的面积
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


def multi_scale_infer():
    shutil.rmtree("test_result", ignore_errors=True)
    args = init_args()

    model = DetInfer(args.model_path)
    tic = time.time()
    for name in tqdm(os.listdir(args.img_path)):
        img_path = os.path.join(args.img_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box_list1, score_list1 = model.predict(img, is_output_polygon=False)

        img1 = np.ascontiguousarray(np.rot90(img))  # 逆时针旋转90°
        box_list2, score_list2 = model.predict(img1, is_output_polygon=False)
        box2 = []
        if len(box_list2) > 0:
            for i in box_list2:
                box2.append(rotation_point(img1, -90, point=i)[1])  # 顺时针旋转回来
        # img = draw_ocr_box_txt(img, box_list)

        img2 = np.ascontiguousarray(np.rot90(img, -1))  # 顺时针旋转90°
        box_list3, score_list3 = model.predict(img2, is_output_polygon=False)
        box3 = []
        if len(box_list3) > 0:
            for i in box_list3:
                box3.append(rotation_point(img1, 90, point=i)[1])  # 逆时针旋转回来

        box_list = box_list1 + box2 + box3
        score_list = score_list1 + score_list2 + score_list3
        # print(np.array(box_list).shape, np.array(score_list).shape)
        keep = py_cpu_pnms(np.array(box_list), np.array(score_list), thresh=0.1)
        # assert len(box_list) == len(score_list)
        box_list = [box_list[i] for i in keep]
        score_list = [score_list[i] for i in keep]
        write_txt_file(box_list, score_list, img_idx=name.split('.')[0].split('_')[-1])
        # print(len(box_list), box_list[0])
        img = draw_bbox(img, box_list)

        os.makedirs('test_result', exist_ok=True)
        img = img[:, :, ::-1]
        cv2.imwrite(filename=f'test_result/result_{name}', img=img)
    print(f'avg infer image in {(time.time() - tic) / len(os.listdir(args.img_path)):.4f}s')


def write_txt_file(bboxs, scores, img_idx: str):
    # print(img_idx)
    txt_dir = 'result_txt'
    # shutil.rmtree(txt_dir, ignore_errors=True)
    os.makedirs(txt_dir, exist_ok=True)
    with open(os.path.join(txt_dir, f'res_img_{img_idx}.txt'), 'w') as f:
        for bbox, score in zip(bboxs, scores):
            b = []
            for i in bbox:
                b.extend([i[0], i[1]])
            b.append(score)
            b = list(map(str, b))
            f.write(','.join(b))
            f.write('\n')


def single_scale_infer():
    shutil.rmtree("test_result", ignore_errors=True)
    args = init_args()

    model = DetInfer(args.model_path)
    tic = time.time()
    for name in tqdm(os.listdir(args.img_path)):
        img_path = os.path.join(args.img_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        box_list, score_list = model.predict(img, is_output_polygon=False)
        # exit(0)
        # img = draw_ocr_box_txt(img, box_list)
        img = draw_bbox(img, box_list)

        os.makedirs('test_result', exist_ok=True)
        # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, :, ::-1]
        cv2.imwrite(filename=f'test_result/result_{name}', img=img)
    print(
        f'avg infer image in {(time.time() - tic) / len(os.listdir(args.img_path)):.4f}s')


def rotate_90_infer():
    shutil.rmtree("test_result", ignore_errors=True)
    args = init_args()

    model = DetInfer(args.model_path)
    tic = time.time()
    for name in tqdm(os.listdir(args.img_path)):
        img_path = os.path.join(args.img_path, name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = np.ascontiguousarray(np.rot90(img))  # 逆时针旋转90°
        box_list2, score_list2 = model.predict(img1, is_output_polygon=False)
        box2 = []
        if len(box_list2) > 0:
            for i in box_list2:
                b = rotation_point(img1, -90, point=i)[1]
                # print(b.shape)
                # print(b)
                box2.append(b)  # 顺时针旋转回来
        # if len(box2) > 0:
        #     # print(box2[0])
        #     print(type(box2[0]))
        #     break
        # box2 = np.array(box2)
        img = draw_bbox(img, box2)

        os.makedirs('test_result', exist_ok=True)
        # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img[:, :, ::-1]
        cv2.imwrite(filename=f'test_result/result_{name}', img=img)
    print(
        f'avg infer image in {(time.time() - tic) / len(os.listdir(args.img_path)):.4f}s')


if __name__ == '__main__':
    multi_scale_infer()
