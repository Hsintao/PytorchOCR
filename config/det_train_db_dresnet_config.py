#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/7/1 4:55 下午
# @Author : Xintao
# ####################rec_train_options 参数说明##########################
# 识别训练参数
# base_lr：初始学习率
# fine_tune_stage:
#     if you want to freeze some stage, and tune the others.
#     ['backbone', 'neck', 'head'], 所有参数都参与调优
#     ['backbone'], 只调优backbone部分的参数
#     后续更新： 1、添加bn层freeze的代码
# optimizer 和 optimizer_step:
#     优化器的配置， 成对
#     example1： 'optimizer'：['SGD'], 'optimizer_step':[],表示一直用SGD优化器
#     example2:  'optimizer':['SGD', 'Adam'], 'optimizer_step':[160]， 表示前[0,160)个epoch使用SGD优化器，
#                [160,~]采用Adam优化器
# lr_scheduler和lr_scheduler_info：
#     学习率scheduler的设置
# ckpt_save_type作用是选择模型保存的方式
#      HighestAcc：只保存在验证集上精度最高的模型（还是在训练集上loss最小）
#      FixedEpochStep： 按一定间隔保存模型
###
from addict import Dict

config = Dict()
config.exp_name = 'DeformResNet_DBNet'
config.train_options = {
    # for train
    'resume_from': f'./output/{config.exp_name}/checkpoint/latest.pth',  # 继续训练地址
    'third_party_name': '',  # 加载paddle模型可选
    # 模型保存地址，log文件也保存在这里
    'checkpoint_save_dir': f"./output/{config.exp_name}/checkpoint",
    'device': 'cuda:0',  # 不建议修改
    'epochs': 1200,
    'fine_tune_stage': ['backbone', 'neck', 'head'],
    'print_interval': 10,  # step为单位
    'val_interval': 20,  # epoch为单位
    # EpochStep：每隔ckpt_save_epoch个epoch保存一个
    'ckpt_save_type': 'EpochStep',
    'ckpt_save_epoch': 10,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效
}

config.SEED = 927
config.optimizer = {
    'type': 'Adam',
    'lr': 0.002,
    'weight_decay': 1e-4,
}

config.model = {
    'type': "DetModel",
    # ResNet or MobileNetV3
    'backbone': {"type": "DResNet", 'layers': 50, 'pretrained': True},
    'neck': {"type": 'FPN', 'out_channels': 256},
    'head': {"type": "DBHead"},
    'in_channels': 3,
}

config.loss = {
    'type': 'DBLoss',
    'alpha': 1,
    'beta': 10
}

config.post_process = {
    'type': 'DBPostProcess',
    'thresh': 0.3,  # 二值化输出map的阈值
    'box_thresh': 0.5,  # 低于此阈值的box丢弃
    'unclip_ratio': 1.5  # 扩大框的比例
}

# for dataset
# ##lable文件
# 存在问题，gt中str-->label 是放在loss中还是放在dataloader中
config.dataset = {
    'train': {
        'dataset': {
            'type': 'TextDataset',
            'data_root': r'/home/wk/xintao/DB/datasets/total_text/train_images',
            # /Volumes/WDSSD/文本检测/自然场景文字检测挑战赛初赛数据/total_text/
            'file': r'',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            # db 预处理，不需要修改
            'pre_processes': [{'type': 'IaaAugment', 'args': [{'type': 'Fliplr', 'args': {'p': 0.5}},
                                                              {'type': 'Affine', 'args': {
                                                                  'rotate': [-10, 10]}},
                                                              {'type': 'Resize', 'args': {'size': [0.5, 3]}}]},
                              {'type': 'EastRandomCropData', 'args': {
                                  'size': [640, 640], 'max_tries': 50, 'keep_ratio':True}},
                              {'type': 'MakeBorderMap', 'args': {
                                  'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7}},
                              {'type': 'MakeShrinkMap', 'args': {'shrink_ratio': 0.4, 'min_text_size': 8}}],
            # 需要从data_dict里过滤掉的key
            'filter_keys': ['img_name', 'texts', 'ignore_tags', 'shape'],
            'ignore_tags': ['*', '###'],
            'img_mode': 'RGB'
        },
        'loader': {
            'type': 'DataLoader',  # 使用torch dataloader只需要改为 DataLoader
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 4,
            'collate_fn': {
                'type': ''
            }
        }
    },
    'eval': {
        'dataset': {
            'type': 'TextDataset',
            'data_root': '/home/wk/xintao/DB/datasets/total_text/test_images',
            'file': r'',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pre_processes': [{'type': 'ResizeShortSize', 'args': {'short_size': 736, 'resize_text_polys': False}}],
            'filter_keys': [],  # 需要从data_dict里过滤掉的key
            'ignore_tags': ['*', '###'],
            'img_mode': 'RGB'
        },
        'loader': {
            'type': 'DataLoader',
            'batch_size': 1,  # 必须为1
            'shuffle': False,
            'num_workers': 1,
            'collate_fn': {
                'type': 'DetCollectFN'
            }
        }
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)