import numpy as np


class Chrom:
    def __init__(self, nodes=None, cars=None, dis=None, k1=100, k2=1, k3=1):
        """
        Initiate the Chrom class.

        :param ndarray nodes: The array of nodes information, (x, y, demand)
        :param ndarray cars: The array of cars information, (capacity, disLimit)
        :param dis:
        :param k1:
        :param k2:
        :param k3:
        """
        self.nodes = nodes
        self.cars = cars
        self.dis = dis
        self.n_node = nodes.shape[0]-1
        self.n_car = cars.shape[0]
        self.mileage = np.zeros(self.n_car)
        self.load = np.zeros(self.n_car)
        self.travel_time = np.zeros(self.n_car)
        self.sum_weight = sum(self.nodes[:, 2])
        self.time = 0.0
        self.length = 0.0
        self.cnt = 0
        self.valid = True

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.gene = np.arange(1, self.n_node+1)
        np.random.shuffle(self.gene)
        index = 0

        # insert 0 as division
        for i in range(self.n_car-1):
            sum_demand = 0
            while True:
                if index >= self.gene.size:
                    break
                sum_demand += self.nodes[self.gene[index], 2]  # demand
                if sum_demand > self.cars[i, 0]:  # capacity
                    break
                index += 1

            self.gene = np.insert(self.gene, index, 0)

        self.update()

    def update(self):
        for i in range(self.mileage.size):
            self.mileage[i] = self.load[i] = self.travel_time[i] = 0

        iCar, last = 0, 0
        for i in range(self.gene.size):
            if self.gene[i] == 0:
                self.mileage[iCar] += self.dis[last, 0]
                # TODO
                iCar += 1
                last = 0
            else:
                self.mileage[iCar] += self.dis[last, self.gene[i]]
                self.travel_time[iCar] += self.mileage[iCar] * self.nodes[self.gene[i], 2]
                self.load[iCar] += self.nodes[self.gene[i], 2]
                last = self.gene[i]
                if self.load[iCar] > self.cars[iCar, 0]:
                    self.valid = False
                    return

        self.mileage[iCar] += self.dis[last, 0]

        self.valid = True

    def mutation(self):
        i = np.random.randint(0, self.gene.size-1)
        j = np.random.randint(0, self.gene.size-1)
        self.gene[i], self.gene[j] = self.gene[j], self.gene[i]

        self.update()
        if not self.valid:
            self.gene[i], self.gene[j] = self.gene[j], self.gene[i]

            self.valid = True
            self.update()

    def fitness(self):
        if not self.valid:
            return 1e8
        # return self.k1*self.time + self.k2*self.length + self.k3*self.cnt
        return sum(self.travel_time) / sum(self.nodes[:, 2])

    def decode(self):
        pass
