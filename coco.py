import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *


def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx


class Thyroid(data.Dataset):
    def __init__(self, root, transform=None, phase='train', inp_name=None):
        self.root = root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        self.get_anno()
        self.num_classes = len(self.cat2idx)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

    def get_anno(self):
        list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
        self.img_list = json.load(open(list_path, 'r'))
        self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        return self.get(item)

    def get(self, item):
        filename = item['filename']
        labels = sorted(item['labels'])
        img = Image.open(os.path.join(self.root, 'data', 'ThyroidImage2021', filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = np.zeros(self.num_classes, np.float32) - 1
        # print(labels)
        target[labels] = 1
        # print('[dataset/coco target]:',target)
        return (img, filename, self.inp), target
