import numpy as np
import time
import heapq

from Algo.VRP import VRP
from Algo.Chrom import Chrom

n_node = 100
n_car = 25


def solve(filename='in.txt'):
    nodes = []
    cars = []
    with open(filename) as f:
        n_node = int(f.readline().strip('\n'))
        for i in range(n_node+1):
            nodes.append([float(x) for x in f.readline().strip('\n').split(" ")])
        f.readline()
        n_car = int(f.readline().strip('\n'))
        for i in range(n_car):
            cars.append([int(x) for x in f.readline().strip('\n').split(" ")])

    nodes = np.array(nodes)
    cars = np.array(cars)
    vrp = VRP(n_node=n_node, n_car=n_car, nodes=nodes, cars=cars)
    vrp.setWeights(100, 1, 1)

    begin_time = time.time()
    best = vrp.solve()
    end_time = time.time()

    car_chosen = heapq.nlargest(10, range(len(best.travel_time)), best.travel_time.__getitem__)

    per_time = sum(best.travel_time[car_chosen]) / sum(best.load[car_chosen])

    fitness = sum(best.load[car_chosen]) / best.sum_weight + (best.dis.max() - per_time) / best.dis.max()

    print(fitness)
    print(sum(best.load[car_chosen])/best.sum_weight)
    print(per_time)

    print("Total time is ", end_time - begin_time)


if __name__ == '__main__':
    solve(filename='../Datasets/Nodes-100-Orders-600.txt')
