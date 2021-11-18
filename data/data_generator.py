import random
import numpy as np


def randomNumVaccinePersonTotal(maxValue, num):
    """生成总和固定的整数序列
    maxValue: 序列总和
    num：要生成的整数个数

    return
    per_all_persons:list,指定 num个接种点各自待接种的人数
    """
    maxValue = int(maxValue)
    suiji_ser = random.sample(range(1, maxValue), k=num - 1)  # 在1~maxValue之间，采集num个数据
    suiji_ser.append(0)  # 加上数据开头
    suiji_ser.append(maxValue)
    suiji_ser = sorted(suiji_ser)
    per_all_persons = [suiji_ser[i] - suiji_ser[i - 1] for i in range(1, len(suiji_ser))]  # 列表推导式，计算列表中每两个数之间的间隔

    return per_all_persons


nodes_list = [[600, 100, 30], [400, 50, 20], [100, 30, 10]]

# ---------------------------------------------
orders_list = randomNumVaccinePersonTotal(100, 30)
orders_list = [order/10 for order in orders_list]
# orders_list = np.random.randint(1, 9, 100)
orders_list = np.insert(orders_list, 0, 0)

nodes = np.random.random((31, 2))

filename = '../Datasets/Nodes-30-Orders-100.txt'
with open(filename, 'w') as f:
    f.write('30')
    for i in range(31):
        s = "\n{:.2f} {:.2f} {:.2f}".format(nodes[i, 0], nodes[i, 1], orders_list[i])
        f.write(s)
    f.write('\n')
    f.write('\n20')
    for i in range(20):
        f.write('\n1 2')

# print('测试结果', randomNumVaccinePersonTotal(600, 100))
