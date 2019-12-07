import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision

from models import M1, M2
from train import train_M1, train_M2

torch.manual_seed(1337)
np.random.seed(1337)


if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    torch.cuda.manual_seed(1337)
else:
    DEVICE = None


PATH = '../data/'

MODELS = {
    'M1': M1,
    'M2': M2
}

TRAINING = {
    'M1': train_M1,
    'M2': train_M2
}



def load_data(n_labeled, batch_size=64):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_labeled = torchvision.datasets.MNIST(PATH, download=True, train=True, transform=transforms)
    train_unlabeled = copy.deepcopy(train_labeled)

    n_classes = np.unique(train_labeled.train_labels)
    n_labeled_class = n_labeled // n_classes

    x_labeled, y_labeled, x_unlabeled, y_unlabeled = map(lambda x: [], range(4))
    for i in range(n_classes):
        mask = train_labeled.train_labels == i
        x_masked = train_labeled.data[mask]
        y_masked = train_labeled.train_labels[mask]
        np.random.shuffle(x_masked)
        x_labeled.append(x_masked[:n_labeled_class])
        x_unlabeled.append(x_masked[n_labeled_class:])
        y_labeled.append(y_masked[:n_labeled_class])
        y_unlabeled.append(y_masked[n_labeled_class:])

    labeled_dataset.train_data = torch.cat(x_labeled)
    labeled_dataset.train_labels = torch.cat(y_labeled)
    unlabeled_dataset.train_data = torch.cat(x_unlabeled)
    unlabeled_dataset.train_labels = torch.cat(y_unlabeled)

    dl_train_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=batch_size, shuffle=True)
    dl_train_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=batch_size, shuffle=True)

    test = torchvision.datasets.MNIST(PATH, download=True, train=False, transform=transforms)
    dl_test = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return dl_train_labeled, dl_train_unlabeled, dl_test



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 'M2')
    args = parser.parse_args()
    dl_train_labeled, dl_train_unlabeled, dl_test = load_data()
    
    model = MODELS[args.model]()
    TRAINING[args.model](model, dl_train_labeled, dl_train_unlabeled, dl_test)
    



if __name__ == '__main__':
    main()