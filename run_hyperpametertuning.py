#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import os, shutil

#Constants
PATH='dataset'
VALUE_ITERATION_MODEL_PATH='saved_models/value-iteration'
RANDOMIZED_MODEL_PATH='saved_models/randomized-algo'
SARSA_MODEL_PATH='saved_models/sarsa'
TD_MODEL_PATH = 'saved_models/td'
QLEARNING_MODEL_PATH = 'saved_models/Q'

######### Perform Hyper-Parameter Tuning #################
hyper_parameters_grid_search = {
    'alpha' : [0.01, 0.99],
    'k' : [4], 
    'discountrate' : [0.01, 0.99],
    'beta_weight' : [0.01, 0.99]
}

def graph_iteration_vs_reward(model, model_load_path, k=4, m=10, with_comparison=False):
    print('Plotting for ', model)
    files = os.listdir(model_load_path)
    files.sort()
    num_rows=len(files)
    row=0
    fig, axs = plt.subplots(num_rows, 1, figsize=(8, 2*num_rows))  # Adjust figsize as needed
    title = model_load_path[13:]
    fig.suptitle(f'Iteration vs Reward for {title}', fontsize=16) 

    for file in files:
        rs = MDP(path='dataset', k=1, save_path=model_load_path)
        rs.load(file)
        k = file[-5]
        y = rs.iteration_vs_reward
        x = [i+1 for i in range(len(y))]
        axs[row].plot(x, y)
        axs[row].set_title(title+',k='+k)
        axs[row].set_xlabel('Iteration')
        axs[row].set_ylabel('Reward')
        row += 1
        np.savez('hyper-parametersgraph/'+model+'_'+k, iteration=x, reward=y)

    plt.tight_layout()
    plt.savefig('hyper-parametersgraph/'+model+'_iteration_vs_reward.png')
    plt.show()
    #Remove from Saved_Models
    shutil.rmtree(model_load_path)

#2D Grid Search
for alpha in hyper_parameters_grid_search['alpha']:
    for discountrate in hyper_parameters_grid_search['discountrate']:
        for beta_weight in hyper_parameters_grid_search['beta_weight']:
            
            # #Hyper-Parameter Tuning for Value Iteration
            k=max(hyper_parameters_grid_search['k'])
            print(f'Hyperparameters Tuning - α={alpha}, β={beta_weight}, γ={discountrate},k={k}')
            saved_file_suffix = f'_α={alpha},β={beta_weight},γ={discountrate}'
            # mixture_model = MixtureModel(path='dataset', k=k, alpha=alpha, beta_weight=beta_weight,verbose=True, save_path=VALUE_ITERATION_MODEL_PATH+saved_file_suffix)
            # mixture_model.generate_model(max_iteration=10000)
            # graph_iteration_vs_reward('value_iteration_'+saved_file_suffix,VALUE_ITERATION_MODEL_PATH+saved_file_suffix, k=k, m=10)
            
            # for i in range(k):
            #     mm = MDP(PATH, k=i+1, save_path=TD_MODEL_PATH+saved_file_suffix)
            #     mm.initialise_mdp()
            #     mm.td_learning_for_optimal_policies(N=100)
            # graph_iteration_vs_reward('td_'+saved_file_suffix,TD_MODEL_PATH+saved_file_suffix, k=k, m=10)
            
            for i in range(k):
                mm = MDP(PATH, k=i+1, save_path=QLEARNING_MODEL_PATH+saved_file_suffix)
                mm.initialise_mdp()
                mm.q_learning_for_optimal_policies(N=100)
            graph_iteration_vs_reward('q_'+saved_file_suffix,QLEARNING_MODEL_PATH+saved_file_suffix, k=k, m=10)


            # #Hyper-Paramter Tuning for Rest of Approaches
            # for k in hyper_parameters_grid_search['k']:
            #     print(f'Hyperparameters Tuning - α={alpha}, β={beta_weight}, γ={discountrate},k={k}')
            #     saved_file_suffix = f'_α={alpha},β={beta_weight},γ={discountrate}'
            #     mm = MDP(PATH, k=k, save_path=RANDOMIZED_MODEL_PATH+saved_file_suffix)
            #     mm.initialise_mdp()
            #     mm.randomized_algorithm_for_optimal_policies(N=100)
            # graph_iteration_vs_reward('randomized_algo_'+saved_file_suffix,RANDOMIZED_MODEL_PATH+saved_file_suffix, k=k, m=10)


