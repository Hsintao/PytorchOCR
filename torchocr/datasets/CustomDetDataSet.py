# /usr/bin/env python
# coding: utf-8
# @File    : CustomDetDataSet.py.py
# @Time    : 2020/6/30
# @Author  : Xintao
import os
import cv2
import json
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchocr.datasets.det_modules import *


class TextDataset(Dataset):
    def __init__(self, config):
        assert config.img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = config.ignore_tags
        # 加载字符级标注
        self.load_char_annotation = False
        self.data_root = config.data_root
        self.data_list = self.load_data()
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = config.img_mode
        self.filter_keys = config.filter_keys
        self._init_pre_processes(config.pre_processes)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std)
        ])

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self) -> list:
        data_list = []

        img_names = os.listdir(self.data_root)
        # if 'train_images' in self.data_root:
        #     gts_root = self.data_root.replace('train_images', 'train_gts')
        # elif 'test_images' in self.data_root:
        #     gts_root = self.data_root.replace('test_images', 'test_gts')
        gts_root = self.data_root.replace('train_images', 'train_gts').replace('test_images', 'test_gts')
        for name in img_names:
            polygons = []
            texts = []
            illegibility_list = []
            language_list = []
            img_path = os.path.join(self.data_root, name)
            gt_path = os.path.join(gts_root, name + '.txt')
            with open(gt_path, 'r', encoding='utf-8') as f:
                infos = [i.strip().split(',') for i in f.readlines()]

            for info in infos:
                # polygons.append(list(map(int, info[:-1])))
                line = list(map(int, info[:-1]))
                tmp = [[line[i * 2], line[i * 2 + 1]] for i in range(int(len(line) // 2))]
                polygons.append(tmp)
                texts.append(info[-1])
                illegibility_list.append(False)
                language_list.append('any')
            data_list.append({'img_path': img_path, 'img_name': name, 'text_polys': np.array(polygons),
                              'texts': texts, 'ignore_tags': illegibility_list})
        return data_list

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        # try:
        data = copy.deepcopy(self.data_list[index])
        im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        data['img'] = im
        data['shape'] = [im.shape[0], im.shape[1]]
        data = self.apply_pre_processes(data)

        if self.transform:
            data['img'] = self.transform(data['img'])
        data['text_polys'] = data['text_polys'].tolist()
        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
            return data_dict
        else:
            return data
        # except:
        #     return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from config.det_train_db_config import config
    from torchocr.utils import show_img, draw_bbox

    from matplotlib import pyplot as plt

    dataset = TextDataset(config.dataset.train.dataset)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        img = data['img']
        shrink_label = data['shrink_map']
        threshold_label = data['threshold_map']

        print(threshold_label.shape, threshold_label.shape, img.shape)
        show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        img = draw_bbox(img[0].numpy().transpose(1, 2, 0), np.array(data['text_polys']))
        show_img(img, title='draw_bbox')
        plt.show()
        pass
