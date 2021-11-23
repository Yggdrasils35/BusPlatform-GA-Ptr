import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm


def sequence_generator():
    points = [i for i in range(1, 8)]
    return itertools.permutations(points)


def get_cost(depot, nodes, sequence):
    """

    :param depot: [1, 2] array the start depot
    :param nodes: [9, 3] array the locations and the weights of all nodes
    :param sequence: [9] list the solution of the certain nodes
    :return:
    """
    cost = 0
    length = 0
    locations = np.append(depot, nodes[:, :2], axis=0)
    weights = nodes[:, 2]
    sequence = list(sequence)

    if sequence[0] != 0:
        sequence.insert(0, 0)
    num = len(sequence)
    for i in range(1, num):
        length += np.linalg.norm(locations[sequence[i]] - locations[sequence[i-1]])
        cost += weights[sequence[i]-1] * length
    return cost


def get_solution(depot, nodes, sequences):
    mincost = np.inf
    best_sequence = []
    for sequence in sequences:
        sequence = list(sequence)
        cost = get_cost(depot, nodes, sequence)
        if cost < mincost:
            mincost = cost
            best_sequence = sequence
    return np.array(best_sequence)


sequences = []
pre_sequences = sequence_generator()
for sequence in pre_sequences:
    sequence = list(sequence)
    sequences.append(sequence)


class TSPDataset(Dataset):
    """
    Random TSP dataset

    """

    def __init__(self, data_size, seq_len, solver=get_solution, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None

        sample = {'Points': tensor, 'Solution': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))

            nodes = np.random.rand(self.seq_len, 2)
            weights = np.random.randint(1, 30, size=(self.seq_len, 1), dtype=int)
            weights[0, 0] = 0
            points = np.append(nodes, weights, axis=1)

            points_list.append(points)
        solutions_iter = tqdm(points_list, unit='solve')

        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i' % (i+1, len(points_list)))
                depot = points[0, :2]
                depot = depot[np.newaxis, :]
                nodes = points[1:]
                solutions.append(get_solution(depot, nodes, sequences))
        else:
            solutions = None

        return {'Points_List': points_list, 'Solutions': solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec


if __name__ == '__main__':
    dataset = TSPDataset(200, 8)
    np.save('./DataSets/TestSet8.npy', dataset)
    print('Success!')