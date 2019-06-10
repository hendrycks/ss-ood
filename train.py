# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
from models.resnet import resnet18
from models.cbam.model_resnet import ResidualNet
import torch.nn.functional as F
import opencv_functional as cv2f
import cv2
import itertools
import torch.utils.model_zoo as model_zoo
import math
import random

parser = argparse.ArgumentParser(description='Trains a one-class model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_class', '-in', type=int, default=0, help='Class to have as the target/in distribution.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/',
                    help='Folder to save checkpoints.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=10, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(1)
np.random.seed(1)

classes = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
           'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover',
           'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile',
           'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']

train_data_in = dset.ImageFolder('./one_class_train/' + classes[args.in_class])
test_data = dset.ImageFolder('./one_class_test/' + classes[args.in_class])

expanded_params = ((0, -56, 56), (0, -56, 56))

shift = np.cumsum([0] + [len(p) for p in expanded_params[:-1]]).tolist()
num_params = [len(expanded_params[i]) for i in range(len(expanded_params))]
n_p1, n_p2 = num_params[0], num_params[1]
output_dim = sum(num_params) + 4  # +4 due to four rotations

pert_configs = []
for tx, ty in itertools.product(*expanded_params):
    pert_configs.append((tx, ty))

num_perts = len(pert_configs)

resize_and_crop = trn.Compose([trn.Resize(256), trn.RandomCrop(224)])


class PerturbDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, train_mode=True):
        self.dataset = dataset
        self.train_mode = train_mode

    def __getitem__(self, index):
        x, _ = self.dataset[index // num_perts]
        pert = pert_configs[index % num_perts]

        x = np.asarray(resize_and_crop(x))

        if np.random.uniform() < 0.5:
            x = x[:, ::-1]
        x = cv2f.affine(np.asarray(x), 0, (pert[0], pert[1]), 1, 0,
                        interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_REFLECT_101)

        label = [expanded_params[i].index(pert[i]) for i in range(len(expanded_params))]
        label = np.vstack((label + [0], label + [1], label + [2], label + [3]))

        x = trnF.to_tensor(x.copy()).unsqueeze(0).numpy()
        x = np.concatenate((x, np.rot90(x, 1, axes=(2, 3)),
                            np.rot90(x, 2, axes=(2, 3)), np.rot90(x, 3, axes=(2, 3))), 0)

        return torch.FloatTensor(x), label

    def __len__(self):
        if self.train_mode:
            return 1300 * num_perts
        else:
            return 100 * num_perts


train_data_in = PerturbDataset(train_data_in, train_mode=True)
test_data = PerturbDataset(test_data, train_mode=False)

train_loader = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
# net = resnet18()
net = ResidualNet('ImageNet', 18, output_dim, 'CBAM')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# /////////////// Training ///////////////


def train():
    net.train()  # enter train mode
    loss_avg = 0.0
    for data, target in train_loader:
        data = data.view(-1, 3, 224, 224)
        target = target.view(data.size(0), -1)
        t1, t2, t3 = target[:, 0], target[:, 1], target[:, 2]
        data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()

        # forward
        x = net(2 * data - 1)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss = (F.cross_entropy(x[:, :n_p1], t1) +
                F.cross_entropy(x[:, n_p1:n_p1 + n_p2], t2) +
                F.cross_entropy(x[:, n_p1 + n_p2:], t3)) / 3.
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    state['train_loss'] = loss_avg


def test():
    loss_avg = 0.0
    net.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 3, 224, 224)
            target = target.view(data.size(0), -1)
            t1, t2, t3 = target[:, 0], target[:, 1], target[:, 2]
            data, t1, t2, t3 = data.cuda(), t1.cuda(), t2.cuda(), t3.cuda()

            # forward
            x = net(2 * data - 1)

            loss = (F.cross_entropy(x[:, :n_p1], t1) +
                    F.cross_entropy(x[:, n_p1:n_p1 + n_p2], t2) +
                    F.cross_entropy(x[:, n_p1 + n_p2:], t3)) / 3.

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

print('Beginning Training\n')

# Main loop
for epoch in range(0, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, classes[args.in_class] + '_' + str(epoch) + '.pt'))
    # Let us not waste space and delete the previous model
    prev_path = os.path.join(args.save, classes[args.in_class] + '_' + str(epoch - 1) + '.pt')
    if os.path.exists(prev_path): os.remove(prev_path)

    # Show results

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'])
    )
