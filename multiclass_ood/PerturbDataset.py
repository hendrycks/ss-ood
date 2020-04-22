# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn_prime import WideResNet
import sklearn.metrics as sk
from PIL import Image
import opencv_functional as cv2f
import cv2
import itertools

import pdb


normalize = trn.Normalize([0.5] * 3, [0.5] * 3)
randomly_crop = trn.RandomCrop(32, padding=4)

class PerturbDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, train_mode=True):
        self.dataset = dataset
        self.num_points = len(self.dataset.data)
        self.train_mode = train_mode

    def __getitem__(self, index):
        x_orig, classifier_target = self.dataset[index]

        if self.train_mode == True and np.random.uniform() < 0.5:
            x_orig = np.copy(x_orig)[:, ::-1]
        else:
            x_orig =  np.copy(x_orig)

        if self.train_mode == True:
            x_orig = Image.fromarray(x_orig)
            x_orig = randomly_crop(x_orig)
            x_orig = np.asarray(x_orig)

        x_tf_0 = np.copy(x_orig)
        x_tf_90 = np.rot90(x_orig.copy(), k=1).copy()
        x_tf_180 = np.rot90(x_orig.copy(), k=2).copy()
        x_tf_270 = np.rot90(x_orig.copy(), k=3).copy()

        possible_translations = list(itertools.product([0, 8, -8], [0, 8, -8]))
        num_possible_translations = len(possible_translations)
        tx, ty = possible_translations[random.randint(0, num_possible_translations - 1)]
        tx_target = {0: 0, 8: 1, -8: 2}[tx]
        ty_target = {0: 0, 8: 1, -8: 2}[ty]
        x_tf_trans = cv2f.affine(np.asarray(x_orig).copy(), 0, (tx, ty), 1, 0, interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_REFLECT_101)

        return \
            normalize(trnF.to_tensor(x_tf_0)), \
            normalize(trnF.to_tensor(x_tf_90)), \
            normalize(trnF.to_tensor(x_tf_180)), \
            normalize(trnF.to_tensor(x_tf_270)), \
            normalize(trnF.to_tensor(x_tf_trans)), \
            torch.tensor(tx_target), \
            torch.tensor(ty_target), \
            torch.tensor(classifier_target)

    def __len__(self):
        return self.num_points

class PerturbDatasetCustom(torch.utils.data.Dataset):
    """
    Used to perturb the custom tensor datasets
    """

    def __init__(self, dataset, train_mode=True):
        self.dataset = dataset
        self.num_points = len(self.dataset.tensors[0])
        assert train_mode == False, "Not Implemented yet."

    def __getitem__(self, index):
        x_orig, classifier_target = self.dataset[index]
        x_orig = x_orig.numpy()
        classifier_target = classifier_target.numpy()

        x_tf_0 = np.copy(x_orig)
        x_tf_90 = np.rot90(x_orig.copy(), k=1).copy()
        x_tf_180 = np.rot90(x_orig.copy(), k=2).copy()
        x_tf_270 = np.rot90(x_orig.copy(), k=3).copy()

        possible_translations = list(itertools.product([0, 8, -8], [0, 8, -8]))
        num_possible_translations = len(possible_translations)
        tx, ty = possible_translations[random.randint(0, num_possible_translations - 1)]
        tx_target = {0: 0, 8: 1, -8: 2}[tx]
        ty_target = {0: 0, 8: 1, -8: 2}[ty]
        x_tf_trans = cv2f.affine(np.asarray(x_orig).copy(), 0, (tx, ty), 1, 0, interpolation=cv2.INTER_CUBIC, mode=cv2.BORDER_REFLECT_101)

        return \
            normalize(trnF.to_tensor(x_tf_0)), \
            normalize(trnF.to_tensor(x_tf_90)), \
            normalize(trnF.to_tensor(x_tf_180)), \
            normalize(trnF.to_tensor(x_tf_270)), \
            normalize(trnF.to_tensor(x_tf_trans)), \
            torch.tensor(tx_target), \
            torch.tensor(ty_target), \
            torch.tensor(classifier_target)

    def __len__(self):
        return self.num_points
