import itertools

import torch
import torch.nn as nn

from .models import M1, M2

OUTPUT_PATH = ''

LR = 0.0003
BETA1 = 0.9
BETA2 = 0.999
ALPHA = 0.1

def train_M1(model: M1, dl_labeled, dl_unlabeled, dl_test, n_epochs, device):
    def loss(x, y, z, p_x_z):
        L = p_x_z.log_prob(x).sum(1) + \
            model.p_y.log_prob(y) + \
                model.p_z.log_prob(z).sum(1)
        return -L
    opt = torch.optim.Adam(model.parameters(), lr=LR, beta=(BETA1, BETA2))
    for epoch in range(n_epochs):
        batches = itertools.chain(iter(dl_labeled), iter(dl_unlabeled))
        for x, _ in batches:
            x = x.to_device(device)
            q_z_x = model.encode_z(x)
            z = q_z_x.rsample()
            p_x_z = model.decode(z)
            x_ = p_x_z.rsample()
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), '../log/')


def train_M2(model: M2, dl_labeled, dl_unlabeled, dl_test, n_epochs):
    def loss_func(x, y, z, p_x_yz, q_z_xy):
        return p_x_yz.log_prob(x).sum(1) + \
            model.p_y.log_prob(y) + \
                model.p_z.log_prob(z).sum(1) - \
                    q_z_xy.log_prob(z).sum(1)

    opt = torch.optim.Adam(model.parameters(), lr=LR, beta=(BETA1, BETA2))
    n_batches = len(dl_labeled) + len(dl_unlabeled)
    dl_labeled_iterable = iter(dl_labeled)
    dl_unlabeled_iterable = iter(dl_unlabeled)

    unlabeled_per_labeled = len(dl_unlabeled) // len(dl_unlabeled) + 1
    
    for epoch in range(n_epochs):
        for batch_id in range(n_batches):
            if batch_id % unlabeled_per_labeled:
                x, y = next(dl_unlabeled_iterable)
                q_y = model.encode_y(x)
                q_z_xy = model.encode_z(x, y)
                z = q_z_xy.rsample()
                p_x_yz = model.decode(y, z)
                loss = loss_func(x, y, z, p_x_yz, q_z_xy)
                loss -= ALPHA * len(dl_labeled) * q_y.log_prob(y)
            else:
                x, y = next(dl_labeled_iterable)
                q_y = model.encode_y(x)
                loss = - q_y.entropy()
                for y in q_y.enumerate_support():
                    q_z_xy = model.encode_z(x, y)
                    z = q_z_xy.rsample()
                    p_x_yz = model.decode(y, z)
                    L = loss_func(x, y, z, model.p_y, model.p_z, p_x_yz, q_z_xy)
                    loss += q_y.log_prob(y).exp() * (-L)

            loss = loss.mean(0)
            opt.zero_grad()
            loss.backward()
            opt.step()

        torch.save(model.state_dict(), '../log/')

def eval_model(model, dl_test):
