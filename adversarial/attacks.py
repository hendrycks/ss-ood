import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tensor_clamp(x, a_min, a_max):
    """
    like torch.clamp, except with bounds defined as tensors
    """
    out = torch.clamp(x - a_max, max=0) + a_max
    out = torch.clamp(out - a_min, min=0) + a_min
    return out


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


def tensor_clamp_l2(x, center, radius):
    """batched clamp of x into l2 ball around center of given radius"""
    x = x.data
    diff = x - center
    diff_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
    project_select = diff_norm > radius
    if project_select.any():
        diff_norm = diff_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        new_x = x
        new_x[project_select] = (center + (diff / diff_norm) * radius)[project_select]
        return new_x
    else:
        return x


class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True, attack_rotations=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.attack_rotations = attack_rotations

    def forward(self, model, bx, by, by_prime, curr_batch_size):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits, pen = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits[:curr_batch_size], by, reduction='sum')
                if self.attack_rotations:
                    loss += F.cross_entropy(model.module.rot_pred(pen[curr_batch_size:]), by_prime, reduction='sum') / 8.
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx

class PGD_l2(nn.Module):
    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        init_noise = normalize_l2(torch.randn(bx.size()).cuda()) * np.random.rand() * self.epsilon
        adv_bx = (bx + init_noise).clamp(0, 1).requires_grad_()

        for i in range(self.num_steps):
            logits = model(adv_bx * 2 - 1)

            loss = F.cross_entropy(logits, by, reduction='sum')

            grad = normalize_l2(torch.autograd.grad(loss, adv_bx, only_inputs=True)[0])
            adv_bx = adv_bx + self.step_size * grad
            adv_bx = tensor_clamp_l2(adv_bx, bx, self.epsilon).clamp(0, 1)
            adv_bx = adv_bx.data.requires_grad_()

        return adv_bx
