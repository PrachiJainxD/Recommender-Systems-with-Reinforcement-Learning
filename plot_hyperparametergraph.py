#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

palette = sns.color_palette("Set2") #sns.color_palette("Spectral", n_colors=num_colors)
color = palette.as_hex() #= ['blue', 'red', 'green', 'pink', 'black', 'magenta', 'cyan', 'gray']
sns.set_style("darkgrid")

def get_x_y(file):
    data = np.load('hyper-parametersgraph/'+file)
    x, y = data['iteration'], data['reward']
    # print(file, x, y)
    return x, y

#Value Iteration
# file = 'value_iteration__α=0.01,β=0.99,γ=0.99_6.npz'
# x1, y1 = get_x_y(file)
# plt.plot(x1, y1, label='α=0.01,β=0.99,γ=0.99,k=6')

# file = 'value_iteration__α=0.99,β=0.01,γ=0.99_6.npz'
# x2, y2 = get_x_y(file)
# plt.plot(x2, y2, label='α=0.99,β=0.01,γ=0.99,k=6')

# file = 'value_iteration__α=0.99,β=0.99,γ=0.99_3.npz'
# x, y = get_x_y(file)
# plt.plot(x, y, label='α=0.99,β=0.99,γ=0.99,k=3')

# plt.xlim(1, 6)
# plt.xlabel('#Iterations', fontsize=14)
# plt.ylabel('Rewards', fontsize=14)
# plt.legend(loc='upper left')
# plt.title('TD Iteration - Hyper-parameter Tuning', fontsize=16)
# plt.show()
# plt.savefig('benchmarks/td_hyperparameters.png')


#TD Learning
# file = 'td__α=0.99,β=0.99,γ=0.99_4.npz'
# x1, y1 = get_x_y(file)

# file = 'td__α=0.01,β=0.01,γ=0.99_2.npz'
# x2, y2 = get_x_y(file)

# file = 'td__α=0.99,β=0.01,γ=0.99_4.npz'
# x, y = get_x_y(file)

# # plt.plot(x1, y1, label='td__α=0.99,β=0.99,γ=0.99,k=4')
# plt.plot(x2, y2, label='td__α=0.01,β=0.01,γ=0.99,k=2')
# plt.plot(x, y, label='td__α=0.99,β=0.01,γ=0.99,k=4')
# plt.ylim(1, 11e6)
# plt.xlabel('#Iterations', fontsize=14)
# plt.ylabel('Rewards', fontsize=14)
# plt.legend(loc='upper left')
# plt.title('TD Learning - Hyper-parameter Tuning', fontsize=16)
# #plt.show()
# plt.savefig('benchmarks/td_hyperparameters.png')
# plt.show()

#Q Learning
file = 'q__α=0.01,β=0.99,γ=0.99_2.npz'
x1, y1 = get_x_y(file)

file = 'q__α=0.99,β=0.01,γ=0.99_2.npz'
x2, y2 = get_x_y(file)

# file = 'td__α=0.99,β=0.01,γ=0.99_4.npz'
# x, y = get_x_y(file)

plt.plot(x1, y1, label='q__α=0.01,β=0.99,γ=0.99_2.npz')
plt.plot(x2, y2, label='q__α=0.99,β=0.01,γ=0.99_2.npz')
# plt.plot(x, y, label='td__α=0.99,β=0.01,γ=0.99,k=4')
#plt.ylim(1, 11e6)
plt.xlabel('#Iterations', fontsize=14)
plt.ylabel('Rewards', fontsize=14)
plt.legend(loc='upper left')
plt.title('Q Learning - Hyper-parameter Tuning', fontsize=16)
#plt.show()
plt.savefig('benchmarks/q_hyperparameters.png')
plt.show()

