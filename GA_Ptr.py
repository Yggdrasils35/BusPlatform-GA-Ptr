import copy
import heapq
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
            cars.append([float(x) for x in f.readline().strip('\n').split(" ")])

    nodes = np.array(nodes)
    cars = np.array(cars)

    dis = np.zeros((n_node + 1, n_node + 1))
    for i in range(n_node + 1):
        for j in range(n_node + 1):
            dis[i, j] = np.sqrt((nodes[i, 0] - nodes[j, 0]) ** 2 + (nodes[i, 1] - nodes[j, 1]) ** 2)

    return nodes, cars, dis


model = PointerNet(embedding_dim=256,
                   hidden_dim=512,
                   gru_layers=3,
                   dropout=0.01,
                   bidir=True)
model.load_state_dict(torch.load('./Parameters/parameterGRU64.pkl'))
model = model.cuda()


class Chrom:
    nodes, cars, dis = data_generate(filename='./Datasets/Nodes-100-Orders-600.txt')
    max_dis = dis.max()
    n_car = cars.shape[0]
    n_node = nodes.shape[0] - 1

    def __init__(self):
        """
        Initiate the Chrom class.

        """
        self.mileage = np.zeros(self.n_car)  # useless
        self.load = np.zeros(self.n_car)  # useless
        self.nodes_in_car = []  # 每辆车的序列
        self.weights_in_car = np.zeros(self.n_car)
        self.fitness_list = [0 for _ in range(self.n_car)]
        self.travel_time = np.zeros(self.n_car) # 每辆车的总时间（length * weight）
        self.sum_weight = sum(self.nodes[:, 2])  # weight即总乘客数
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

        self.cars_changed = []  # 表明每次mutation改变的车号

        nodes_features = [self.nodes[_nodes, :] for _nodes in self.nodes_in_car]

        # 将不规则list用0填充
        length = max(map(len, nodes_features))
        batch_nodes_arr = np.zeros((self.used_cars, length, 3))
        for i in range(self.used_cars):
            batch_nodes_arr[i, :, :] = np.vstack((nodes_features[i], np.repeat(self.nodes[0][np.newaxis, :], length-len(nodes_features[i]), axis=0)))
        batch_nodes = np.array(batch_nodes_arr)

        with torch.no_grad():
            batch_nodes = torch.tensor(batch_nodes, dtype=torch.float32, requires_grad=False).cuda()
            o, solutions = model(batch_nodes)

        solutions = solutions.cpu().numpy()

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

        self.valid = True

    def mutation(self):
        i, j = np.random.choice(range(self.n_car), 2, replace=False)
        is_additional_car = False
        if i >= self.used_cars:
            j = np.random.randint(0, self.used_cars)
            self.used_cars += 1
            is_additional_car = True
        elif j >= self.used_cars:
            i = np.random.randint(0, self.used_cars)
            self.used_cars += 1
            is_additional_car = True

        t_nodes_i, t_nodes_j = copy.deepcopy(self.nodes_in_car[i]), copy.deepcopy(self.nodes_in_car[j])
        length_i = np.random.randint(0, len(self.nodes_in_car[i]))
        length_j = np.random.randint(0, len(self.nodes_in_car[j]))

        i_begin = np.random.randint(1, len(self.nodes_in_car[i]) - length_i + 1)
        j_begin = np.random.randint(1, len(self.nodes_in_car[j]) - length_j + 1)

        i_seg = self.nodes_in_car[i][i_begin:i_begin+length_i]
        j_seg = self.nodes_in_car[j][j_begin:j_begin+length_j]

        del self.nodes_in_car[i][i_begin:i_begin+length_i]
        del self.nodes_in_car[j][j_begin:j_begin+length_j]

        self.nodes_in_car[i] += j_seg
        self.nodes_in_car[j] += i_seg

        self.cars_changed = [i, j]

        self.update()
        if not self.valid:
            self.nodes_in_car[i], self.nodes_in_car[j] = t_nodes_i, t_nodes_j

            self.valid = True
            if is_additional_car:
                self.used_cars -= 1

    def fitness(self):
        if not self.valid:
            return 1e8
        car_chosen = heapq.nlargest(10, range(len(self.fitness_list)), self.fitness_list.__getitem__)
        per_time = sum(self.travel_time[car_chosen]) / (sum(self.weights_in_car[car_chosen]) * self.sum_weight)
        # per_time = sum(self.travel_time)/self.sum_weight
        fitness = sum(self.weights_in_car[car_chosen])*2/3 + 4/3*(self.max_dis - per_time) / self.max_dis
        return fitness

    def decode(self):
        pass

    def getTime(self, car_number, sequence):
        per_car_time = 0
        length = 0
        node_idx = np.array(self.nodes_in_car[car_number])
        nodes = self.nodes[node_idx]
        locations = nodes[:, :2]
        weights = nodes[:, 2]
        sequence = sequence.tolist()

        if sequence[0] != 0:
            sequence.insert(0, 0)
        num = len(sequence)
        for i in range(1, num):
            length += np.linalg.norm(locations[sequence[i]] - locations[sequence[i - 1]])
            per_car_time += weights[sequence[i]] * length

        self.travel_time[car_number] = per_car_time  # 不确定是否有效
        self.weights_in_car[car_number] = sum(weights) / self.sum_weight  # 满足率
        max_avg_travel_time = length
        if max_avg_travel_time == 0:
            self.fitness_list[car_number] = 0
        else:
            self.fitness_list[car_number] = self.weights_in_car[car_number]*10 + (max_avg_travel_time - self.travel_time[car_number]/sum(weights)) / max_avg_travel_time


class VRP:
    def __init__(self):
        self.depot = np.array([0.11, 0.09, 0.00])  # TODO 写的更优美一点

    def solve(self):
        all_fitness = []
        fullfillments = []
        avg_travel_time_list = []

        chroms = []
        while len(chroms) < 1000:
            chrom = Chrom()
            chroms.append(chrom)

        cnt = 0
        numGeneration = 0
        best = Chrom()
        print(best.max_dis)

        while True:
            numGeneration += 1
            chroms.sort(key=lambda x: x.fitness())

            if chroms[0].fitness() < best.fitness():
                best = chroms[0]
                cnt = 0
            else:
                cnt += 1

            if cnt >= 800:
                break

            x = len(chroms) // 2
            for i in range(x, len(chroms)):
                chroms[i] = copy.deepcopy(chroms[i - x])

                chroms[i].mutation()

            cars_to_solve = []
            for i in range(x, len(chroms)):
                cars_to_solve.append(chroms[i].nodes_in_car[chroms[i].cars_changed[0]])
                cars_to_solve.append(chroms[i].nodes_in_car[chroms[i].cars_changed[1]])

            # 将不规则list用0填充
            length = max(map(len, cars_to_solve))
            nodes_in_batch_arr = np.zeros((len(cars_to_solve), length, 3))
            nodes_index = [car+[0]*(length - len(car)) for car in cars_to_solve]  # 不足的补0

            for i in range(len(cars_to_solve)):
                nodes_in_batch_arr[i, :, :] = chroms[i].nodes[nodes_index[i]]

            with torch.no_grad():
                nodes_in_batch = torch.tensor(nodes_in_batch_arr, dtype=torch.float, requires_grad=False).cuda()
                o, p = model(nodes_in_batch)

            del o
            p = p.cpu().numpy()

            for i in range(x, len(chroms)):
                _solution_1 = p[2 * (i - 500)]
                car_num_1 = chroms[i].cars_changed[0]
                solution_1 = _solution_1[_solution_1 < len(chroms[i].nodes_in_car[car_num_1])]

                chroms[i].getTime(car_num_1, solution_1)

                _solution_2 = p[2 * (i - 500) + 1]
                car_num_2 = chroms[i].cars_changed[1]
                solution_2 = _solution_2[_solution_2 < len(chroms[i].nodes_in_car[car_num_2])]

                chroms[i].getTime(car_num_2, solution_2)

                chroms[i].all_time = sum(chroms[i].travel_time) / chroms[i].sum_weight

            car_chosen = heapq.nlargest(10, range(len(best.fitness_list)), best.fitness_list.__getitem__)
            per_time = sum(best.travel_time[car_chosen]) / (sum(best.weights_in_car[car_chosen]) * best.sum_weight)
            fullfillment = sum(best.weights_in_car[car_chosen])
            fitness = fullfillment + (best.max_dis - per_time) / best.max_dis

            all_fitness.append(fitness)
            fullfillments.append(fullfillment)
            avg_travel_time_list.append(per_time)

            print("Generation {} fitness {}".format(numGeneration, best.fitness()))

        # filename = 'data/CVRP-100-per.txt'
        # iteration = len(all_fitness)

        # with open(filename, 'w') as f:
        #     f.write(str(iteration))
        #     for i in range(iteration):
        #         f.write('\n{:.4f} {:.4f} {:.4f}'.format(all_fitness[i], fullfillments[i], avg_travel_time_list[i]))


if __name__ == '__main__':
    vrp = VRP()

    begin_time = time.time()
    vrp.solve()
    end_time = time.time()

    print(end_time - begin_time)
