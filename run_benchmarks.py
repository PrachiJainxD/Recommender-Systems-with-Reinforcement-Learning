from mdp import MDP
import matplotlib.pyplot as plt
import numpy as np

color = ['blue', 'red', 'green', 'pink', 'black', 'magenta', 'cyan', 'gray']

def graph_recommendation_score(model_load_path, scale=4, m=10, with_comparison=False):
    """
    Function to generate a graph for the recommendation score over a range of m for a set of k
    :param scale: the limit to which k should vary
    :param m: a parameter in recommendation score computation
    :param with_comparison: plot a random policy's graph
    :return: None
    """

    fig = plt.figure()
    k = [i for i in range(1, scale+1)]
    x = [i for i in range(m+1)]

    for j in k:
        y_recommendation = []
        y_recommendation_rand = []
        rs = MDP(path='dataset', k=j, save_path='saved_models/'+model_load_path)
        rs.load('mdp-model_k=' + str(j) + '.pkl')
        for i in x:
            if with_comparison:
                rs.initialise_mdp()
                y_recommendation_rand.append(rs.evaluate_recommendation_score(m=i))
            y_recommendation.append(rs.evaluate_recommendation_score(m=i))

        plt.plot(x, y_recommendation, color=color[j-1], label="MC model " + str(j))
        #plt.scatter(x, y_recommendation, color=color[j-1])

        if with_comparison:
            plt.plot(x, y_recommendation_rand, color=(0.2, 0.8, 0.6, 0.6), label="Random model, For m=" + str(m))
            plt.scatter(x, y_recommendation_rand)

        plt.xticks(x)
        plt.yticks([i for i in range(20, 100, 10)])

        # for x1, y in zip(x, y_recommendation):
        #     text = '%.2f' % y
        #     plt.text(x1, y, text)

        if with_comparison:
            for x1, y in zip(x, y_recommendation_rand):
                text = '%.2f' % y
                plt.text(x1, y, text)

    fig.suptitle('Recommendation Score vs Prediction List size')
    plt.xlabel('Prediction List size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('benchmarks/recommendation_score.png')
    plt.show()

def graph_decay_score(model_load_path, scale=3, rand=False):
    """
    Function to generate a graph for the exponential decay score over a range of k
    :param scale: the limit to which k should vary
    :param rand: to use a random policy or not
    :return: None
    """

    fig = plt.figure()
    x = [i + 1 for i in range(scale)]
    y_decay = []
    for i in x:
        rs = MDP(path='dataset', k=i, save_path='saved_models/'+model_load_path)

        if rand:
            rs.initialise_mdp()
            y_decay.append(rs.evaluate_decay_score())
            continue

        rs.load('mdp-model_k=' + str(i) + '.pkl')
        y_decay.append(rs.evaluate_decay_score())

    plt.bar(x, y_decay, width=0.5, color=(0.2, 0.4, 0.6, 0.6))
    xlocs = [i + 1 for i in range(0, 10)]
    for i, v in enumerate(y_decay):
        plt.text(xlocs[i] - 0.46, v + 0.9, '%.2f' % v)

    plt.xticks(x)
    plt.yticks([i for i in range(0, 100, 10)])

    fig.suptitle('Avg Exponential Decay Score vs Number of items in each state')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.savefig('benchmarks/exp_decay_score.png')
    plt.show()

def graph_iteration_vs_reward(model_load_path, k=4, m=10, with_comparison=False):
    rows = 4
    cols = 2

    fig, axs = plt.subplots(rows, cols, figsize=(10, 5*rows))

    for idx in range(k):
        rs = MDP(path='dataset', k=idx+1, save_path='saved_models/'+model_load_path)
        rs.load('mdp-model_k=' + str(idx+1) + '.pkl')
        y = rs.iteration_vs_reward
        x = [i+1 for i in range(len(y))]
        
        row_idx = idx // cols
        col_idx = idx % cols
        #print(idx, row_idx, col_idx)

        axs[row_idx, col_idx].plot(x, y, color=color[idx], label="MC model " + str(idx+1))
        axs[row_idx, col_idx].set_title(f'Iteration vs Reward (k={idx+1})')
        axs[row_idx, col_idx].set_xlabel('Iteration')
        axs[row_idx, col_idx].set_ylabel('Reward')
        axs[row_idx, col_idx].legend()

    plt.tight_layout()
    plt.savefig('benchmarks/iteration_vs_reward_subplots_randomized_algo.png')
    plt.show()

# graph_recommendation_score('mixture-models', scale=8)
# graph_decay_score('mixture-models', scale=8, rand=False)
# graph_iteration_vs_reward('mixture-models', k=8)
graph_iteration_vs_reward('randomized-algo', k=8)