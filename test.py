import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from torchvision.models import resnet18
from models.cbam.model_resnet import ResidualNet
from PIL import Image as PILImage
from tqdm import tqdm
import opencv_functional as cv2f
import cv2
import itertools
import math
import sklearn.metrics as sk

parser = argparse.ArgumentParser(description='Evaluates a one-class model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--in_class', '-in', type=int, default=0, help='Class to have in-distribution.')
parser.add_argument('--test_bs', type=int, default=128)
# Loading details
parser.add_argument('--load', '-l', type=str, default='./snapshots/',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=10, help='Pre-fetching threads.')
args = parser.parse_args()

classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
           'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
           'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
           'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']

test_data = dset.ImageFolder('./one_class_test/' + classes[args.in_class])

expanded_params = ((0, -56, 56), (0, -56, 56), range(4))

shift = np.cumsum([0] + [len(p) for p in expanded_params[:-1]]).tolist()
num_params = [len(expanded_params[i]) for i in range(len(expanded_params))]
n_p1, n_p2, n_p3 = num_params[0], num_params[1], num_params[2]
output_dim = sum(num_params)  # 3 + 3 + 4

pert_configs = []
for tx, ty, k_rotate in itertools.product(*expanded_params):
    pert_configs.append((tx, ty, k_rotate))
num_perts = len(pert_configs)

resize_and_crop = trn.Compose([trn.Resize(256), trn.CenterCrop(224)])


class PerturbDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.pert_number = 0

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        pert = pert_configs[self.pert_number]

        x = np.asarray(resize_and_crop(x))

        x = cv2f.affine(np.asarray(x), 0, (pert[0], pert[1]), 1, 0,
                        interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_REFLECT_101)
        x = np.rot90(x, pert[2])

        return trnF.to_tensor(x.copy()), [expanded_params[i].index(pert[i]) for i in range(len(expanded_params))]

    def __len__(self):
        return len(self.dataset)


test_data = PerturbDataset(test_data)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
# net = resnet18(num_classes=output_dim)
net = ResidualNet('ImageNet', 18, output_dim, 'CBAM')

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):

        model_name = os.path.join(args.load, classes[args.in_class] + '_' + str(i) + '.pt')

        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i, 'Make sure this is the model you want restored.')
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////

print('\nUsing', classes[args.in_class], 'as typical data')

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader):
    _score = []

    scores_across_transforms = np.zeros(len(loader.dataset))
    with torch.no_grad():
        for t in range(num_perts):
            loader.dataset.pert_number = t
            for i, (data, (t1, t2, t3)) in enumerate(loader):
                data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()

                output = net(2 * data - 1)

                smax1 = F.softmax(output[:, :n_p1], 1)
                smax2 = F.softmax(output[:, n_p1:n_p1 + n_p2], 1)
                smax3 = F.softmax(output[:, n_p1 + n_p2:], 1)

                mask1 = torch.zeros_like(smax1)
                mask2 = torch.zeros_like(smax2)
                mask3 = torch.zeros_like(smax3)

                mask1.scatter_(1, t1.view(-1, 1), 1.)
                mask2.scatter_(1, t2.view(-1, 1), 1.)
                mask3.scatter_(1, t3.view(-1, 1), 1.)

                score = (smax1 * mask1).sum(1) + (smax2 * mask2).sum(1) + (smax3 * mask3).sum(1)

                scores_across_transforms[i * args.test_bs:(i + 1) * args.test_bs] += to_np(score)

    return -scores_across_transforms.copy()


in_score = get_ood_scores(test_loader)

# /////////////// End Detection Prelims ///////////////

# /////////////// OOD Detection ///////////////
auroc_list = []


def get_auroc(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc


def get_and_print_results(ood_loader):
    out_score = get_ood_scores(ood_loader)
    auroc = get_auroc(out_score, in_score)
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))


# /////////////// Held-out Classes ///////////////

datasets = []
for class_name in classes:
    if class_name != classes[args.in_class]:
        datasets.append(dset.ImageFolder('../one_class_test/' + class_name))

ood_data = PerturbDataset(torch.utils.data.ConcatDataset(datasets))

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=False,
                                         num_workers=args.prefetch, pin_memory=True)

get_and_print_results(ood_loader)
