import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('tableau-colorblind10')

all_fitness = []

GA_all_fitness = []

filename = 'Results/convergence.txt'
with open(filename) as f:
    all_length = int(f.readline().strip('\n'))
    for i in range(all_length):
        line = f.readline().strip('\n')
        fitness = float(line)
        all_fitness.append(fitness)

filename = 'Results/CVRP-100-GA.txt'
with open(filename) as f:
    length = int(f.readline().strip('\n'))
    for i in range(length):
        line = f.readline().strip('\n')
        fitness = float(line)
        GA_all_fitness.append(fitness)


plt.figure()
# plt.title('Fitness')
plt.plot(range(all_length), all_fitness, range(length), GA_all_fitness)
plt.xlabel('iterations')
plt.ylabel('Traveling Time')
plt.legend(['Our method', 'Pure GA'])
plt.savefig('../data/Figures/Convergence.png', dpi=1200)
plt.show()
