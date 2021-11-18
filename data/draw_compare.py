import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# sns.set()
plt.style.use('tableau-colorblind10')
print(plt.style.available)

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

GA_Ptr = [[1.1704, 0.4950, 0.4343],
          [1.1942, 0.4875, 0.3755],
          [1.5099, 1.0000, 0.5166]]

GA = [[0.8544, 0.4467, 0.7925],
      [0.9602, 0.4475, 0.6240],
      [1.1843, 0.6800, 0.5226]]

RL = [[0.4850, 0.565],
      [0.4775, 0.595],
      [1.0000,  0.520]]

df_time = pd.DataFrame([[0.5166, 0.5066, 0.5226],
                        [0.3755, 0.5952, 0.6240],
                        [0.4343, 0.5650, 0.7925]
                        ],
                       index=['30', '50', '100'],
                       columns=pd.Index(['Our method', 'L2i', 'GA']))

df_fulfillment = pd.DataFrame([[1.0000, 1.0000, 0.6800],
                               [0.9750, 0.9550, 0.8950],
                               [0.9900, 0.9700, 0.8934]
                               ],
                              index=['30', '50', '100'],
                              columns=pd.Index(['Our method', 'L2i', 'GA']))

df_fulfillment.plot.bar()
plt.xlabel('Stations Numbers')
plt.ylabel('Fulfillment of Orders')
plt.savefig('./Figures/Fulfillment.png', dpi=1200)
plt.show()

