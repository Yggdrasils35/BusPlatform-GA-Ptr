import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import argparse

from models.PointerNet import PointerNet

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

    # Data
    parser.add_argument('--train_size', default=2000, type=int, help='Training data size')
    parser.add_argument('--val_size', default=100, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=100, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # TSP
    parser.add_argument('--nof_points', type=int, default=10, help='Number of points in TSP')
    # Network
    parser.add_argument('--embedding_size', type=int, default=256, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--nof_grus', type=int, default=3, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

    params = parser.parse_args()

    if params.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_grus,
                       params.dropout,
                       params.bidir)

    print('Loading model...')
    model.load_state_dict(torch.load('./Parameters/parameterGRU64.pkl'))
    print('Loaded finished!')

    if USE_CUDA:
        model = model.cuda()

    # dataset = np.load('./DataSets/TestSet.npy', allow_pickle=True)
    #
    # dataloader = DataLoader(dataset,
    #                         batch_size=32,
    #                         shuffle=True,
    #                         num_workers=0)

    test_points = np.random.random((10, 10, 3))

    test_batch_1 = torch.tensor(test_points, dtype=torch.float).cuda()
    test_batch_2 = torch.tensor(test_points[:2], dtype=torch.float).cuda()

    o, p = model(test_batch_2)
    print(p)

    o, p = model(test_batch_1)
    print(p)


if __name__ == '__main__':
    main()
