from mdp import MDP
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_colors = 8
palette = sns.color_palette("Set2") #sns.color_palette("Spectral", n_colors=num_colors)
color = palette.as_hex() #= ['blue', 'red', 'green', 'pink', 'black', 'magenta', 'cyan', 'gray']

def graph_recommendation_score(model_load_path, filename, label, scale=4, m=10, with_comparison=False):
    """
    Function to generate a graph for the recommendation score over a range of m for a set of k
    :param scale: the limit to which k should vary
    :param m: a parameter in recommendation score computation
    :param with_comparison: plot a random policy's graph
    :return: None
    """
    sns.set_style("darkgrid")
    fig = plt.figure()
    k = [i for i in range(1, scale+1)]
    x = [i for i in range(m+1)]

    for j in (k):
        y_recommendation = []
        y_recommendation_rand = []
        rs = MDP(path='dataset', k=j, save_path='saved_models/'+model_load_path)
        rs.load(filename + str(j) + '.pkl')
        for i in x:
            if with_comparison:
                rs.initialise_mdp()
                y_recommendation_rand.append(rs.evaluate_recommendation_score(m=i))
            y_recommendation.append(rs.evaluate_recommendation_score(m=i))

        plt.plot(x, y_recommendation, color=color[j-1], label=label + str(j))
        #plt.scatter(x, y_recommendation, color=color[j-1])

        if with_comparison:
            plt.plot(x, y_recommendation_rand, color=(0.2, 0.8, 0.6, 0.6), label="Random model, For m=" + str(m))
            plt.scatter(x, y_recommendation_rand)

        plt.xticks(x)
        plt.yticks([i for i in range(10, 100, 10)])

        # for x1, y in zip(x, y_recommendation):
        #     text = '%.2f' % y
        #     plt.text(x1, y, text)

        if with_comparison:
            for x1, y in zip(x, y_recommendation_rand):
                text = '%.2f' % y
                plt.text(x1, y, text)

    fig.suptitle('Recommendation Score vs Top i Recommendations', fontsize=14)
    plt.xlabel('Top i Recommendations', fontsize=12)
    plt.ylabel('Recommendation Score', fontsize=12)
    plt.legend()
    plt.savefig('benchmarks/recommendation_score.png')
    plt.show()

def graph_decay_score(model_load_path, filename, scale=3, rand=False):
    """
    Function to generate a graph for the exponential decay score over a range of k
    :param scale: the limit to which k should vary
    :param rand: to use a random policy or not
    :return: None
    """

    fig = plt.figure()
    sns.set_style("darkgrid")
    x = [i + 1 for i in range(scale)]
    y_decay = []
    for i in (x):
        rs = MDP(path='dataset', k=i, save_path='saved_models/'+model_load_path)

        if rand:
            rs.initialise_mdp()
            y_decay.append(rs.evaluate_decay_score())
            continue

        rs.load(filename + str(i) + '.pkl')
        y_decay.append(rs.evaluate_decay_score())

    plt.bar(x, y_decay, width=0.5, color= color, alpha=0.7)
    xlocs = [i + 1 for i in range(0, 10)]
    for i, v in enumerate(y_decay):
        plt.text(xlocs[i] - 0.16, v + 0.9, '%.2f' % v)

    plt.xticks(x)
    plt.yticks([i for i in range(0, 100, 10)])

    fig.suptitle('Avg Exponential Decay Score vs Last K Purchases', fontsize = 12)
    plt.xlabel('K', fontsize = 12)
    plt.ylabel('Exponential Decay Score', fontsize = 12)
    plt.savefig('benchmarks/exp_decay_score.png')
    plt.show()

def graph_iteration_vs_reward(model_load_path, filename, label, k=4, m=10, with_comparison=False):
    rows = k//2
    cols = 2

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5*rows))
    for idx in range(k):
        rs = MDP(path='dataset', k=idx+1, save_path='saved_models/'+model_load_path)
        rs.load(filename + str(idx+1) + '.pkl')
        y = rs.iteration_vs_reward
        x = [i+1 for i in range(len(y))]
        row_idx = idx // cols
        col_idx = idx % cols
        #print(idx, row_idx, col_idx)

        axs[row_idx, col_idx].plot(x, y, color=color[idx], label=label+ str(idx+1))
        axs[row_idx, col_idx].fill_between(x, y, color=color[idx], alpha=0.3)
        axs[row_idx, col_idx].set_title(f'Iterations vs Rewards (k={idx+1})')
        axs[row_idx, col_idx].set_xlabel('# Iterations')
        axs[row_idx, col_idx].set_ylabel('Rewards')
        axs[row_idx, col_idx].legend()

    plt.tight_layout()
    plt.savefig('benchmarks/'+model_load_path+'_iteration_vs_reward.png')
    plt.show()


k_value = 4
# graph_recommendation_score('td', filename ='td_mdp-model_k=', label ="TD Learning ", scale=k_value)
# graph_decay_score('td',  filename ='td_mdp-model_k=', scale=k_value, rand=False)
# graph_iteration_vs_reward('td', filename ='td_mdp-model_k=', label ="TD Learning ", k=k_value)

graph_recommendation_score('value-iteration', filename ='mdp-model_k=', label ="value-iteration", scale=k_value)
graph_decay_score('value-iteration',  filename ='mdp-model_k=', scale=k_value, rand=False)
graph_iteration_vs_reward('value-iteration', filename ='mdp-model_k=', label ="value-iteration", k=k_value)
