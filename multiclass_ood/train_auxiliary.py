"""
Train a classifier with auxiliary self-supervision
THIS MAY NOT HAVE BEEN THE EXACT CODE, BUT IF MEMORY SERVES CORRECTLY, IT IS
"""

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

from PerturbDataset import PerturbDataset

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Train a classifier with auxiliary self-supervision',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--in-class', type=int, default=None)

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--rot-loss-weight', type=float, required=True, help='Multiplicative factor on the rot losses')
parser.add_argument('--transl-loss-weight', type=float, required=True, help='Multiplicative factor on the translation losses')

# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/TEMP', help='Folder to save checkpoints.')

# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

if os.path.exists(os.path.join(args.save)):
    # Ask if we want to overwrite
    response = input("Save path {0} exists. Continue? [yes/no]: ".format(args.save))
    if response != 'yes':
        print("Exiting")
        sys.exit(0)
else:
    os.makedirs(args.save, )

# Save the command we used to run this
if os.path.isfile(os.path.join(args.save, 'training_command.txt')):
    os.remove(os.path.join(args.save, 'training_command.txt'))

with open(os.path.join(args.save, 'training_command.txt'), 'w') as f:
    f.write(str(state))

torch.manual_seed(1)
np.random.seed(1)


def main():

    print("Using CIFAR 10")
    train_data_in = dset.CIFAR10('~/datasets/cifarpy', train=True, download=True)
    test_data = dset.CIFAR10('~/datasets/cifarpy', train=False, download=True)
    num_classes = 10

    # 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    # Must do != None to make sure 0 case works
    if args.in_class != None:
        print("Removing all but class {0} from train dataset and test dataset".format(args.in_class))
        train_data_in.data = train_data_in.data[train_data_in.targets == args.in_class*np.ones_like(train_data_in.targets)]
        test_data.data = test_data.data[test_data.targets == args.in_class*np.ones_like(test_data.targets)]
    else:
        print("Keeping all classes in both train/test datasets")

    train_data_in = PerturbDataset(train_data_in, train_mode=True)
    test_data = PerturbDataset(test_data, train_mode=False)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.prefetch,
        pin_memory=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=False
    )

    # Create model
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    net.x_trans_head = nn.Linear(128, 3)
    net.y_trans_head = nn.Linear(128, 3)
    net.rot_head = nn.Linear(128, 4)

    # Get GPUs ready
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    # Set up optimization stuffs
    optimizer = torch.optim.SGD(
        net.parameters(),
        state['learning_rate'],
        momentum=state['momentum'],
        weight_decay=state['decay'],
        nesterov=True
    )

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader_in),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate
        )
    )

    print('Beginning Training\n')

    # Main loop
    for epoch in range(0, args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()

        train(net, state, train_loader_in, optimizer, lr_scheduler)
        test(net, state, test_loader)

        # Save model
        torch.save(
            net.state_dict(),
            os.path.join(
                args.save,
                'layers_{0}_widenfactor_{1}_inclass_{2}_transform_trflossweight_{3}_epoch_{4}.pt'.format(
                    str(args.layers),
                    str(args.widen_factor),
                    str(args.in_class),
                    str(args.rot_loss_weight) + "_" + str(args.transl_loss_weight),
                    str(epoch)
                )
            )
        )

        # Let us not waste space and delete the previous model
        prev_path = os.path.join(
            args.save,
            'layers_{0}_widenfactor_{1}_inclass_{2}_transform_trflossweight_{3}_epoch_{4}.pt'.format(
                str(args.layers),
                str(args.widen_factor),
                str(args.in_class),
                str(args.rot_loss_weight) + "_" + str(args.transl_loss_weight),
                str(epoch - 1)
            )
        )
        if os.path.exists(prev_path):
            os.remove(prev_path)

        # Show results

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Accuracy {4:.3f}%'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['test_loss'],
            state['test_accuracy'] * 100
        ))

def train(net, state, train_loader_in, optimizer, lr_scheduler):
    net.train()  # enter train mode
    loss_avg = 0.0
    for x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, target_class in tqdm(train_loader_in, dynamic_ncols=True):
        batch_size = x_tf_0.shape[0]
        
        # Sanity check
        assert x_tf_0.shape[0] == \
            x_tf_90.shape[0] == \
            x_tf_180.shape[0] == \
            x_tf_270.shape[0] == \
            x_tf_trans.shape[0] == \
            target_trans_x.shape[0] == \
            target_trans_y.shape[0] == \
            target_class.shape[0]

        batch = np.concatenate((
            x_tf_0,
            x_tf_90,
            x_tf_180,
            x_tf_270,
            x_tf_trans
        ), 0)
        batch = torch.FloatTensor(batch).cuda()

        target_rots = torch.cat((
            torch.zeros(batch_size),
            torch.ones(batch_size),
            2 * torch.ones(batch_size),
            3 * torch.ones(batch_size)
        ), 0).long()

        lr_scheduler.step()
        optimizer.zero_grad()

        # Forward together
        logits, pen = net(batch)

        # Calculate losses
        if args.ngpu > 1:
            classification_logits = logits[:batch_size]
            rot_logits            = net.module.rot_head(pen[:4*batch_size])
            x_trans_logits        = net.module.x_trans_head(pen[4*batch_size:])
            y_trans_logits        = net.module.y_trans_head(pen[4*batch_size:])
        else:
            classification_logits = logits[:batch_size]
            rot_logits            = net.rot_head(pen[:4*batch_size])
            x_trans_logits        = net.x_trans_head(pen[4*batch_size:])
            y_trans_logits        = net.y_trans_head(pen[4*batch_size:])

        classification_loss = F.cross_entropy(classification_logits, target_class.cuda())
        rot_loss = F.cross_entropy(rot_logits, target_rots.cuda()) * args.rot_loss_weight
        x_trans_loss = F.cross_entropy(x_trans_logits, target_trans_x.cuda()) * args.transl_loss_weight
        y_trans_loss = F.cross_entropy(y_trans_logits, target_trans_y.cuda()) * args.transl_loss_weight

        loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)

        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.9 + float(loss) * 0.1

    state['train_loss'] = loss_avg

def test(net, state, test_loader):
    loss_avg = 0.0
    net.eval()
    with torch.no_grad():
        correct = 0
        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, target_class in test_loader:
            batch_size = x_tf_0.shape[0]
            assert x_tf_0.shape[0] == \
                x_tf_90.shape[0] == \
                x_tf_180.shape[0] == \
                x_tf_270.shape[0] == \
                x_tf_trans.shape[0] == \
                target_trans_x.shape[0] == \
                target_trans_y.shape[0] == \
                target_class.shape[0]

            batch = np.concatenate((
                x_tf_0,
                x_tf_90,
                x_tf_180,
                x_tf_270,
                x_tf_trans
            ), 0)
            batch = torch.FloatTensor(batch).cuda()

            target_rots = torch.cat((
                torch.zeros(batch_size),
                torch.ones(batch_size),
                2 * torch.ones(batch_size),
                3 * torch.ones(batch_size)
            ), 0).long()

            # Forward
            logits, penultimate = net(batch)

            # Calculate losses
            if args.ngpu > 1:
                classification_logits = logits[:batch_size]
                rot_logits            = net.module.rot_head(penultimate[:4*batch_size])
                x_trans_logits        = net.module.x_trans_head(penultimate[4*batch_size:])
                y_trans_logits        = net.module.y_trans_head(penultimate[4*batch_size:])
            else:
                classification_logits = logits[:batch_size]
                rot_logits            = net.rot_head(penultimate[:4*batch_size])
                x_trans_logits        = net.x_trans_head(penultimate[4*batch_size:])
                y_trans_logits        = net.y_trans_head(penultimate[4*batch_size:])

            classification_loss = F.cross_entropy(classification_logits, target_class.cuda())
            rot_loss = F.cross_entropy(rot_logits, target_rots.cuda()) * args.rot_loss_weight
            x_trans_loss = F.cross_entropy(x_trans_logits, target_trans_x.cuda()) * args.transl_loss_weight
            y_trans_loss = F.cross_entropy(y_trans_logits, target_trans_y.cuda()) * args.transl_loss_weight

            loss = classification_loss + ((rot_loss + x_trans_loss + y_trans_loss) / 3.0)

            # accuracy
            pred = classification_logits.cpu().data.max(1)[1]
            correct += pred.eq(target_class.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

if __name__ == "__main__":
    main()
