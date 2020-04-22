"""
Test OOD with one classifier network
Metric used is the maximum softmax probability


Best tests so far:
 - Both use classification_loss + total_rot_loss as the anomaly score for now. 
   Check git commit where this was added.
With CIFAR-100 and Gaussian as OOD:
    saurav@pinwheel9:~/nnets-ood$ python3 test_auxiliary_ood_NEW.py --model=snapshots/train_aux_new_rot_only/layers_40_widenfactor_2_inclass_None_transform_trflossweight_1.0_0.0_epoch_199.pt
    AUROC = 0.93108209
    saurav@pinwheel9:~/nnets-ood$ python3 test_auxiliary_ood_NEW.py --model=snapshots/vanilla_classifier/cifar10_wrn_baseline_epoch_99.pt --vanilla
    AUROC = 0.9201875500000001

With only CIFAR-100 as OOD:
    saurav@pinwheel9:~/nnets-ood$ python3 test_auxiliary_ood_NEW.py --model=snapshots/train_aux_new_rot_only/layers_40_widenfactor_2_inclass_None_transform_trflossweight_1.0_0.0_epoch_199.pt
    AUROC = 0.89920049
    saurav@pinwheel9:~/nnets-ood$ python3 test_auxiliary_ood_NEW.py --model=snapshots/vanilla_classifier/cifar10_wrn_baseline_epoch_99.pt --vanilla
    AUROC = 0.8730536999999999
"""
import copy
import numpy as np
import sys
import os
import pickle
import argparse
import itertools
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn_prime import WideResNet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random
import pdb
import gc # Garbage collector
import opencv_functional as cv2f
import cv2
import time
from PerturbDataset import PerturbDataset, PerturbDatasetCustom

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    description='Test OOD with one classifier network. Metric used is the maximum softmax probability',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Test setup
parser.add_argument('--test_bs', type=int, default=500)
parser.add_argument('--in_dataset', choices=["CIFAR10", "CIFAR100"], default="CIFAR10")
parser.add_argument('--out_dataset', choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
parser.add_argument('--prefetch', type=int, default=10, help='Pre-fetching threads.')

# Loading details
parser.add_argument('--architecture', type=str, default='wrn', choices=['wrn'], help='Choose architecture.')
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--model', '-l', type=str, required=True, help='Trained PyTorch model')
parser.add_argument('--vanilla', action='store_true')

# Test-time training
parser.add_argument('--test-time-train', action='store_true')
parser.add_argument('--test-epochs', type=int, default=None)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

assert not (args.test_time_train and (args.test_epochs == None)), "Muse give --test-epochs when --test-time-train is on."

def get_tensors(gpu_only=True):
    """
    For debugging purposes
    """
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass

def kl_div(d1, d2):
    """
    Compute KL-Divergence between d1 and d2.
    """
    dirty_logs = d1 * torch.log2(d1 / d2)
    return torch.sum(torch.where(d1 != 0, dirty_logs, torch.zeros_like(d1)), axis=1)


def jsd(d1, d2):
    """
    Calculate Jensen-Shannon Divergence between d1 and d2
    Square-root this to get the Jensen-Shannon distance
    """
    M = 0.5 * (d1 + d2)
    return 0.5 * kl_div(d1, M) + 0.5 * kl_div(d2, M)


def get_anomaly_scores(net, loader):
    """
    Return the anomaly score for each element in the loader using net.
    """
    net = net.cuda()
    net = net.eval()
    scores = []
    with torch.no_grad():
        for x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, _ in tqdm(loader):
            scores.append(get_anomaly_score_batch(
                x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, net
            ))

    net = net.cpu()
    scores = torch.cat(scores, dim=0)
    assert len(scores) == 10000
    assert len(scores.shape) == 1
    
    return scores.cpu().numpy()


def get_anomaly_score_batch(x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, net):
    batch_size = x_tf_0.shape[0]
    assert x_tf_0.shape[0] == \
        x_tf_90.shape[0] == \
        x_tf_180.shape[0] == \
        x_tf_270.shape[0] == \
        x_tf_trans.shape[0] == \
        target_trans_x.shape[0] == \
        target_trans_y.shape[0]

    # pdb.set_trace()

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
    classification_smax = F.softmax(logits[:batch_size], dim=1)
    rot_smax            = F.softmax(net.rot_head(penultimate[:4*batch_size]), dim=1)
    # x_trans_smax        = F.softmax(net.x_trans_head(penultimate[4*batch_size:]), dim=1)
    # y_trans_smax        = F.softmax(net.y_trans_head(penultimate[4*batch_size:]), dim=1)

    class_uniform_dist = torch.ones_like(classification_smax) * 0.1

    # classification_loss, _ = torch.max(classification_smax, dim=1)
    # classification_loss = classification_loss * -1.0

    classification_loss = -1 * kl_div(class_uniform_dist, classification_smax)

    rot_one_hot = torch.zeros_like(rot_smax).scatter_(1, target_rots.unsqueeze(1).cuda(), 1)
    rot_loss = kl_div(rot_one_hot, rot_smax)

    # x_trans_one_hot = torch.zeros_like(x_trans_smax).scatter_(1, target_trans_x.unsqueeze(1).cuda(), 1)
    # x_trans_loss = kl_div(x_trans_one_hot, x_trans_smax)

    # y_trans_one_hot = torch.zeros_like(y_trans_smax).scatter_(1, target_trans_y.unsqueeze(1).cuda(), 1)
    # y_trans_loss = kl_div(y_trans_one_hot, y_trans_smax)

    rot_0_loss, rot_90_loss, rot_180_loss, rot_270_loss = torch.chunk(rot_loss, 4, dim=0)

    total_rot_loss = (rot_0_loss + rot_90_loss + rot_180_loss + rot_270_loss) / 4.0

    # Max of all rot losses
    # all_rot_losses = torch.cat([rot_0_loss.unsqueeze(1), rot_90_loss.unsqueeze(1), rot_180_loss.unsqueeze(1), rot_270_loss.unsqueeze(1)], dim=1)
    # total_rot_loss, _ = torch.max(all_rot_losses, dim=1)

    # pdb.set_trace()

    # Use these weights to weight the sum of classification loss vs total_rot_loss
    # both_losses_per_item = torch.cat([total_rot_loss.unsqueeze(1), classification_loss.unsqueeze(1)], dim=1)
    # weights = F.softmax(both_losses_per_item, dim=1)
    # curr_score = torch.sum(weights * both_losses_per_item, dim=1)

    curr_score = classification_loss + total_rot_loss
    
    # OOM Clean up
    batch.cpu()
    target_rots.cpu()
    logits.cpu()
    penultimate.cpu()
    classification_smax.cpu()
    rot_smax.cpu()
    # x_trans_smax.cpu()
    # y_trans_smax.cpu()
    class_uniform_dist.cpu()
    classification_loss.cpu()
    rot_one_hot.cpu()
    rot_loss.cpu()
    # x_trans_one_hot.cpu()
    # x_trans_loss.cpu()
    # y_trans_one_hot.cpu()
    # y_trans_loss.cpu()
    rot_0_loss.cpu()
    rot_90_loss.cpu()
    rot_180_loss.cpu()
    rot_270_loss.cpu()
    total_rot_loss.cpu()

    del batch
    del target_rots
    del logits
    del penultimate
    del classification_smax
    del rot_smax
    # del x_trans_smax
    # del y_trans_smax
    del class_uniform_dist
    del classification_loss
    del rot_one_hot
    del rot_loss
    # del x_trans_one_hot
    # del x_trans_loss
    # del y_trans_one_hot
    # del y_trans_loss
    del rot_0_loss
    del rot_90_loss
    del rot_180_loss
    del rot_270_loss
    del total_rot_loss

    return curr_score


# def get_anomaly_scores_TTT(net, loader):
#     """
#     Return the anomaly score for each element in the loader using net.
#     Uses reverse test-time training
#     """
#     net = net.eval()
    
#     scores = []
#     num_batches_done = 0
#     for x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, _ in loader:
#         batch_size = x_tf_0.shape[0]
#         print("Getting scores for batch {0} / {1}".format(num_batches_done, len(loader)))
#         for i in tqdm(range(batch_size)):
#             scores.append(get_single_anomaly_score_TTT(
#                 x_tf_0[i], x_tf_90[i], x_tf_180[i], x_tf_270[i], x_tf_trans[i], target_trans_x[i], target_trans_y[i], net
#             ).cpu())
#         num_batches_done += 1
#         break # Only do one batch for now.

#     scores = torch.cat(scores, dim=0)
#     # Commented out for testing
#     # assert len(scores) == 10000 
#     assert len(scores.shape) == 1
    
#     return scores.cpu().numpy()


# def  get_single_anomaly_score_TTT(x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, net):
#     """
#     Get the anomaly score for the single example passed in using Test-Time training
#     Insipred by https://arxiv.org/pdf/1909.13231.pdf
#     """
#     test_net = copy.deepcopy(net) # Net that we will train with this singular example

#     for p in test_net.parameters():
#         p.requires_grad = True

#     # Make it look like a batch size of 1
#     x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y = \
#         x_tf_0.unsqueeze(0), \
#         x_tf_90.unsqueeze(0), \
#         x_tf_180.unsqueeze(0), \
#         x_tf_270.unsqueeze(0), \
#         x_tf_trans.unsqueeze(0), \
#         target_trans_x.unsqueeze(0), \
#         target_trans_y.unsqueeze(0)
    
#     test_net.train().cuda()
    
#     batch = np.concatenate((
#         x_tf_0,
#         x_tf_90,
#         x_tf_180,
#         x_tf_270,
#         x_tf_trans
#     ), 0)
#     batch = torch.FloatTensor(batch).cuda()

#     target_rots = torch.cat((
#         torch.zeros(1),
#         torch.ones(1),
#         2 * torch.ones(1),
#         3 * torch.ones(1)
#     ), 0).long()

#     test_optimizer = torch.optim.SGD(
#         test_net.parameters(),
#         0.001, # Learning Rate
#         momentum=0, # Turn off these like https://arxiv.org/pdf/1909.13231.pdf
#         weight_decay=0,
#         nesterov=False
#     )

#     # SSL Test-time training
#     for test_epoch in range(args.test_epochs):
#         test_optimizer.zero_grad()
#         logits, pen = test_net(batch)
#         rot_logits  = test_net.rot_head(pen[:4])
#         rot_smax    = F.softmax(rot_logits, dim=1)
#         rot_loss    = -1 * torch.sum(kl_div(rot_smax, torch.ones_like(rot_smax) * 0.25), dim=0)
        
#         rot_loss.backward()
#         test_optimizer.step()


#     # Now we use the net
#     test_net.eval()
#     for p in test_net.parameters():
#         p.requires_grad = False
#     score_for_this_example = get_anomaly_score_batch(x_tf_0, x_tf_90, x_tf_180, x_tf_270, x_tf_trans, target_trans_x, target_trans_y, test_net).cpu()

#     # Catch-all to fix OOM Errors (??)
#     batch.cpu()
#     target_rots.cpu()

#     del batch
#     del target_rots

#     del test_optimizer
    
#     test_net.cpu()
#     del test_net
#     torch.cuda.empty_cache()

#     return score_for_this_example


def get_anomaly_scores_vanilla_msp(net, loader):
    """
    Return max softmax probability for each example in loader
    """
    net = net.cuda()
    net = net.eval()

    probs = []
    with torch.no_grad():
        for x_tf_0, _, _, _, _, _, _, _ in tqdm(loader):
            x_tf_0 = x_tf_0.cuda()
            
            # The vanilla network was trained with this 2 * batch - 1 thing
            logits, _ = net(2 * x_tf_0 - 1)

            smaxed = F.softmax(logits, dim=1)
            maxes, _  = torch.max(smaxed, dim=1)

            maxes = maxes.cpu()
            probs.append(maxes)

    net = net.cpu()
    probs = torch.cat(probs, dim=0)
    assert probs.shape[0] == 10000
    assert len(probs.shape) == 1

    return -1 * probs.numpy()

def main():
    # Load the model
    net = None
    if args.architecture == 'wrn':
        net = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)
    else:
        raise NotImplementedError()

    if not args.vanilla:
        net.x_trans_head = nn.Linear(128, 3)
        net.y_trans_head = nn.Linear(128, 3)
        net.rot_head = nn.Linear(128, 4)

    if os.path.isfile(args.model):
        net.load_state_dict(torch.load(args.model))
    else:
        raise Exception("Cannot find {0}".format(args.model))
    
    in_data = PerturbDataset(dset.CIFAR10('~/datasets/cifarpy', train=False, download=True), train_mode=False)
    out_data = PerturbDataset(dset.CIFAR100('~/datasets/cifarpy', train=False, download=True), train_mode=False)

    in_loader = torch.utils.data.DataLoader(
        in_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=False
    )

    out_loader = torch.utils.data.DataLoader(
        out_data,
        batch_size=args.test_bs,
        shuffle=False,
        num_workers=args.prefetch,
        pin_memory=False
    )

    if args.vanilla:
        anomaly_func = get_anomaly_scores_vanilla_msp
    else:
        if args.test_time_train:
            raise RuntimeError("--test-time-train doesn't work very well.")
            anomaly_func = get_anomaly_scores_TTT
        else:
            anomaly_func = get_anomaly_scores

    print("Getting anomaly scores for the in_dist set")
    in_probs = anomaly_func(net, in_loader)

    print("Getting anomaly scores for the out_dist set")
    out_probs = anomaly_func(net, out_loader)

    print("Getting anomaly scores for Gaussian data")
    gauss_num_examples = 10000
    dummy_targets = torch.ones(gauss_num_examples)
    gauss_data = torch.from_numpy(np.float32(np.clip(np.random.normal(size=(gauss_num_examples, 32, 32, 3), scale=0.5, loc=0.5), 0, 1)))
    gauss_data = PerturbDatasetCustom(torch.utils.data.TensorDataset(gauss_data, dummy_targets), train_mode=False)
    gauss_loader = torch.utils.data.DataLoader(gauss_data, batch_size=args.test_bs, shuffle=True, num_workers=args.prefetch, pin_memory=False)
    # pdb.set_trace()
    gauss_probs = anomaly_func(net, gauss_loader)

    ground_truths = [0 for _ in range(10000)] 
    ground_truths += [1 for _ in range(10000)] 
    # ground_truths += [1 for _ in range(gauss_num_examples)]

    scores = np.concatenate([
        in_probs, 
        out_probs, 
    #    gauss_probs
    ])

    AUROC = roc_auc_score(ground_truths, scores)
    print("AUROC = {0}".format(AUROC))




if __name__ == "__main__":
    main()
