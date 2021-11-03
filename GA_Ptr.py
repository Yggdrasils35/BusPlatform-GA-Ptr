import copy
import gc
import numpy as np
import time
import torch
from models.PointerNet import PointerNet


def data_generate(filename='in.txt'):
    nodes = []
    cars = []
    with open(filename) as f:
        n_node = int(f.readline().strip('\n'))
        for i in range(n_node + 1):
            nodes.append([float(x) for x in f.readline().strip('\n').split(" ")])
        f.readline()
        n_car = int(f.readline().strip('\n'))
        for i in range(n_car):
            cars.append([int(x) for x in f.readline().strip('\n').split(" ")])

    nodes = np.array(nodes)
    cars = np.array(cars)

    dis = np.zeros((n_node + 1, n_node + 1))
    for i in range(n_node + 1):
        for j in range(n_node + 1):
            dis[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2)

    return nodes, cars, dis


k1 = 100
k2 = 1
k3 = 1

model = PointerNet(embedding_dim=256,
                   hidden_dim=512,
                   gru_layers=3,
                   dropout=0.01,
                   bidir=True)
model.load_state_dict(torch.load('./Parameters/parameterGRU64.pkl'))
model = model.cuda()


class Chrom:
    nodes, cars, dis = data_generate()
    n_car = cars.shape[0]
    n_node = nodes.shape[0] - 1

    def __init__(self):
        """
        Initiate the Chrom class.

        """
        self.mileage = np.zeros(self.n_car)
        self.load = np.zeros(self.n_car)
        self.nodes_in_car = []
        self.travel_time = [0 for _ in range(self.n_car)]
        self.sum_weight = sum(self.nodes[:, 2])
        self.all_time = 0
        self.valid = True
        self.used_cars = 0

        self.gene = np.arange(1, self.n_node + 1)
        np.random.shuffle(self.gene)
        index = 0

        for i in range(self.n_car):
            d_sum = 0
            per_car = [0]
            while True:
                if index >= self.gene.size:
                    break
                d_sum += self.nodes[self.gene[index], 2]  # demand
                if d_sum > self.cars[i, 0]:
                    break
                per_car.append(self.gene[index])
                index += 1
            self.nodes_in_car.append(per_car)
            if not per_car == [0]:
                self.used_cars += 1

        self.cars_changed = []

        nodes_features = [self.nodes[_nodes, :] for _nodes in self.nodes_in_car]

        length = max(map(len, nodes_features))
        batch_nodes_arr = np.zeros((self.used_cars, length, 3))
        for i in range(self.used_cars):
            batch_nodes_arr[i, :, :] = np.vstack((nodes_features[i], np.repeat(self.nodes[0][np.newaxis, :], length-len(nodes_features[i]), axis=0)))
        batch_nodes = np.array(batch_nodes_arr)
        # matrix = np.array([node+[0, 0, 0]*(length-len(node)) for node in nodes_features])
        batch_nodes = torch.tensor(batch_nodes, dtype=torch.float32).cuda()
        o, solutions = model(batch_nodes)
        for i in range(self.used_cars):
            solution = solutions[i][solutions[i] < nodes_features[i].shape[0]]
            self.getTime(i, solution)
        self.all_time = sum(self.travel_time) / self.sum_weight
        self.update()

    def update(self):
        """ To test whether the change is valid """
        if not self.cars_changed:
            return

        for index in self.cars_changed:
            load = 0
            for node in self.nodes_in_car[index]:
                load += self.nodes[node, 2]
                if load > self.cars[index, 0]:
                    self.valid = False
                    return
            # self.travel_time[index] = model(self.nodes_in_car[index])

        self.valid = True

    def mutation(self):
        i, j = np.random.choice(range(self.n_car), 2, replace=False)

        length = np.random.randint(1, min(len(self.nodes_in_car[i]), len(self.nodes_in_car[j])))
        i_begin = np.random.randint(1, len(self.nodes_in_car[i]) - length + 1)
        j_begin = np.random.randint(1, len(self.nodes_in_car[j]) - length + 1)

        self.nodes_in_car[i][i_begin:i_begin + length], self.nodes_in_car[j][j_begin:j_begin + length] = \
            self.nodes_in_car[j][j_begin:j_begin + length], self.nodes_in_car[i][i_begin:i_begin + length]

        self.cars_changed = [i, j]

        self.update()
        if not self.valid:
            self.nodes_in_car[i][i_begin:i_begin + length], self.nodes_in_car[j][j_begin:j_begin + length] = \
                self.nodes_in_car[j][j_begin:j_begin + length], self.nodes_in_car[i][i_begin:i_begin + length]
            self.valid = True

    def fitness(self):
        if not self.valid:
            return 1e8
        return self.all_time

    def decode(self):
        pass

    def getTime(self, car_number, sequence):
        per_car_time = 0
        length = 0
        node_idx = np.array(self.nodes_in_car[car_number])
        nodes = self.nodes[node_idx]
        locations = nodes[:, :2]
        weights = nodes[:, 2]
        sequence = list(sequence)

        if sequence[0] != 0:
            sequence.insert(0, 0)
        num = len(sequence)
        for i in range(1, num):
            length += np.linalg.norm(locations[sequence[i]] - locations[sequence[i - 1]])
            per_car_time += weights[sequence[i] - 1] * length
        if self.all_time < 0:
            print('Wrong')
        self.travel_time[car_number] = per_car_time


class VRP:
    def __init__(self):
        self.depot = np.array([14.36, 8.22, 0])

    def solve(self):

        chroms = []
        while len(chroms) < 1000:
            chrom = Chrom()
            chroms.append(chrom)

        cnt = 0
        numGeneration = 0
        best = Chrom()

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
                chroms[i] = copy.copy(chroms[i - x])

                chroms[i].mutation()

            cars_to_solve = []
            for i in range(x, len(chroms)):
                cars_to_solve.append(chroms[i].nodes_in_car[chroms[i].cars_changed[0]])
                cars_to_solve.append(chroms[i].nodes_in_car[chroms[i].cars_changed[1]])

            length = max(map(len, cars_to_solve))
            nodes_in_batch_arr = np.zeros((len(cars_to_solve), length, 3))
            nodes_index = [car+[0]*(length - len(car)) for car in cars_to_solve]
            for i in range(len(cars_to_solve)):
                nodes_in_batch_arr[i, :, :] = chroms[i].nodes[nodes_index[i]]
            nodes_in_batch = torch.tensor(nodes_in_batch_arr, dtype=torch.float).cuda()
            o, p = model(nodes_in_batch)
            del o
            p = p.data.cpu().numpy()
            gc.collect()
            for i in range(x, len(chroms)):
                solution = p[2 * (i - 500)]
                car_num_1 = chroms[i].cars_changed[0]
                solution_1 = solution[solution < len(chroms[i].nodes_in_car[car_num_1])]
                chroms[i].getTime(car_num_1, solution_1)
                solution = p[2 * (i - 500) + 1]
                car_num_2 = chroms[i].cars_changed[1]
                solution_2 = solution[solution < len(chroms[i].nodes_in_car[car_num_2])]
                chroms[i].getTime(car_num_2, solution_2)

                chroms[i].all_time = sum(chroms[i].travel_time) / chroms[i].sum_weight

            print("Generation {} fitness {}".format(numGeneration, best.fitness()))


if __name__ == '__main__':
    vrp = VRP()

    begin_time = time.time()
    vrp.solve()
    end_time = time.time()

    print(end_time - begin_time)
