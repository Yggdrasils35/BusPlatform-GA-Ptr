import heapq

import numpy as np
import copy
from Algo.Chrom import Chrom


class VRP:
    def __init__(self, n_node=0, n_car=0, nodes=None, cars=None):
        self.n_node = n_node
        self.n_car = n_car
        self.nodes = nodes
        self.cars = cars
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0

    def addNode(self, x, y, demand):
        self.nodes.append((x, y, demand))

    def addCar(self, capacity, disLimit):
        self.cars.append((capacity, disLimit))

    def setWeights(self, k1, k2, k3):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def listToArray(self):
        self.nodes = np.array(self.nodes)
        self.cars = np.array(self.cars)

    def solve(self):
        dis = np.zeros((self.n_node+1, self.n_node+1))
        for i in range(self.n_node+1):
            for j in range(self.n_node+1):
                dis[i, j] = np.sqrt((self.nodes[i, 0] - self.nodes[j, 0])**2 + (self.nodes[i, 1] - self.nodes[j, 1])**2)

        max_dis = dis.max()
        chroms = []

        all_fitness = []
        fullfillments = []
        avg_travel_time_list = []

        while len(chroms) < 1000:
            chrom = Chrom(nodes=self.nodes,
                          cars=self.cars,
                          dis=dis)
            chroms.append(chrom)

        cnt = 0
        numGeneration = 0
        best = Chrom(nodes=self.nodes, cars=self.cars, dis=dis)

        while True:
            numGeneration += 1
            chroms.sort(key=lambda x: x.fitness())

            if chroms[0].fitness() < best.fitness():
                best = chroms[0]
                cnt = 0
            else:
                cnt += 1

            if cnt >= 1000:
                break

            x = len(chroms) // 2
            for i in range(x, len(chroms)):
                chroms[i] = copy.deepcopy(chroms[i - x])

                chroms[i].mutation()

            # car_chosen = heapq.nlargest(10, range(len(best.travel_time)), best.travel_time.__getitem__)
            #
            # per_time = sum(best.travel_time[car_chosen]) / sum(best.load[car_chosen])
            # fullfillment = sum(best.load[car_chosen]) / best.sum_weight
            # fitness = fullfillment + (max_dis - per_time) / max_dis
            fitness = best.fitness()
            #
            all_fitness.append(fitness)
            # fullfillments.append(fullfillment)
            # avg_travel_time_list.append(per_time)

            print("Generation {} fitness {}".format(numGeneration, fitness))

            # if fitness > 0.81:
            #     print(sum(best.load[car_chosen])/best.sum_weight)
            #     print(per_time)

        filename = 'CVRP-100-GA.txt'
        iteration = len(all_fitness)

        # with open(filename, 'w') as f:
        #     f.write(str(iteration))
        #     for i in range(iteration):
        #         f.write('\n{:.4f} {:.4f} {:.4f}'.format(all_fitness[i], fullfillments[i], avg_travel_time_list[i]))
        with open(filename, 'w') as f:
            f.write(str(iteration))
            for i in range(iteration):
                f.write('\n{:.4f}'.format(all_fitness[i]))
        return best
