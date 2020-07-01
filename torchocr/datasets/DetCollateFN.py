# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 14:16
# @Author  : zhoujun
import PIL
import numpy as np
import torch

__all__ = ['DetCollectFN']


class DetCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        # print(to_tensor_keys)
        for k in to_tensor_keys:
            # print(data_dict[k].shape)
            # data_dict[k] = [torch.from_numpy(i) for i in data_dict[k]]
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict
