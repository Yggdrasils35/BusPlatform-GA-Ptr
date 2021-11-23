"""
using TestSet(200 data)
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse

from models.PointerNet import PointerNet
from utils.Data_generator import get_cost

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

    dataset = np.load('./DataSets/TestSet.npy', allow_pickle=True)

    test_num = 200
    dataloader = DataLoader(dataset,
                            batch_size=test_num,
                            shuffle=True,
                            num_workers=0)

    points = []
    solutions = []
    solutions_opt = []
    arr = np.array(list(range(1, 10)))
    opt_list = []
    test_list = []
    random_list = []
    error_list = []

    for i_batch, sample_batched in enumerate(dataloader):
        # solution.append(sample_batched['Solution'])
        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        o, p = model(train_batch)

        solutions = np.array(p)  # get the tested solutions

        # get the points for the test
        for i in range(test_num):
            point_tensor = train_batch[i]
            solution_tensor = list(target_batch[i])
            solution_tensor.insert(0, 0)
            solutions_opt.append(solution_tensor)
            points_list = []
            for point in point_tensor:
                points_list.append(list(point))
            points.append(points_list)
        points = np.array(points)
        solutions_opt = np.array(solutions_opt)

    for i in range(test_num):
        point = points[i]
        point[:, 2] *= 30
        depot = point[0, :2]
        depot = depot[np.newaxis, :]
        nodes = point[1:]
        solution = solutions[i]
        weights = point[:, 2]
        cost = get_cost(depot, nodes, solution)

        solution_opt = solutions_opt[i]
        cost_opt = get_cost(depot, nodes, solution_opt)

        solution_random = np.random.permutation(arr)
        solution_random = np.pad(solution_random, (1, 0))
        cost_random = get_cost(depot, nodes, solution_random)  # random solution

        error_opt = (cost - cost_opt) / cost_opt * 100
        error_list.append(error_opt)

        cost_list = [cost_random, cost_opt, cost]
        opt_list.append(cost_opt)
        test_list.append(cost)
        random_list.append(cost_random)

        print('Test{0}:'.format(i + 1))
        print(solution, 'cost is ', cost, '(Test solution)')
        print(solution_opt, 'cost is ', cost_opt, '(Optimized solution)')
        print(solution_random, 'cost is ', cost_random, '(Random solution)')
        print('The cost error is {0:.2f}%'.format(error_opt), '\n')

        if i % 20 == 0:
            plt.figure(i, (7, 7))

            plt.subplot(221)
            plt.title('Optimized solution')
            plt.scatter(point[:, 0], point[:, 1], s=weights)
            plt.plot(point[solution_opt][:, 0], point[solution_opt][:, 1], 'r')

            plt.subplot(222)
            plt.title('Test solution')
            plt.scatter(point[:, 0], point[:, 1], s=weights)
            plt.plot(point[solution][:, 0], point[solution][:, 1], 'b')

            plt.subplot(223)
            plt.title('Random solution')
            plt.scatter(point[:, 0], point[:, 1], s=weights)
            plt.plot(point[solution_random][:, 0], point[solution_random][:, 1], 'y')

            plt.subplot(224)
            plt.title('cost Comparison')
            plt.barh(range(3), cost_list, tick_label=['Rand', 'Opt', 'Test'], height=0.3)

    plt.figure(figsize=(7, 5))
    plt.title('Total result')
    x = range(1, test_num+1)
    plt.plot(x, opt_list, '-', x, test_list, '-', x, random_list, '-')

    plt.figure(figsize=(7, 5))
    plt.title('Total result')
    x = range(1, test_num + 1)
    plt.plot(x, error_list, '.-')

    error_arr = np.array(error_list)
    sum = error_arr == 0
    print(error_arr[sum].shape)
    print(np.mean(error_list))
    plt.show()


if __name__ == '__main__':
    main()