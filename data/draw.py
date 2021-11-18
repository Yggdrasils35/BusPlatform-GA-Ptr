import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

all_fitness = []
fulfillments = []
avg_travel_time_list = []

GA_all_fitness = []
GA_fulfillments = []
GA_avg_travel_time_list = []

filename = 'Results/Order-400-Car-10.txt'
with open(filename) as f:
    all_length = int(f.readline().strip('\n'))
    for i in range(all_length):
        line = f.readline().strip('\n').split(" ")
        fitness, fulfillment, avg_time = line
        all_fitness.append(float(fitness))
        fulfillments.append(float(fulfillment))
        avg_travel_time_list.append(float(avg_time))

filename = 'Results/Order-400-Car-10-GA.txt'
with open(filename) as f:
    length = int(f.readline().strip('\n'))
    for i in range(length):
        line = f.readline().strip('\n').split(" ")
        fitness, fulfillment, avg_time = line
        GA_all_fitness.append(float(fitness))
        GA_fulfillments.append(float(fulfillment))
        GA_avg_travel_time_list.append(float(avg_time))

x = 800
plt.figure()
plt.title('Average Travel Time')
plt.plot(range(x), avg_travel_time_list[:x], range(x), GA_avg_travel_time_list[:x])
plt.legend(['GA Ptr', 'Pure GA'])

plt.figure()
plt.title('Fulfillment')
plt.plot(range(x), fulfillments[:x], range(x), GA_fulfillments[:x])
plt.legend(['GA Ptr', 'Pure GA'])

plt.figure()
plt.title('Fitness')
plt.plot(range(x), all_fitness[:x], range(x), GA_all_fitness[:x])
plt.legend(['GA Ptr', 'Pure GA'])
plt.show()