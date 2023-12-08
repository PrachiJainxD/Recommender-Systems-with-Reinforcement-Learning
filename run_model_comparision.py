from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

#Constants
PATH='dataset'
MONTE_CARLO_MODEL_PATH='saved_models/monte-carlo'
RANDOMIZED_MODEL_PATH='saved_models/randomized-algo'
K_LAST_PURCHASES = 4

## Load Policy for Mixture Models Monte Carlo ##
def recommendation_score_policy_k(scale, model_load_path):
    fig = plt.figure()
    k = [i for i in range(1, scale+1)]
    y_recommendation = []
    for j in k:
        rs = MDP(path='dataset', k=j, save_path=model_load_path)
        rs.initialise_mdp()
        rs.load('mdp-model_k=' + str(j) + '.pkl')
        y_recommendation.append(rs.evaluate_recommendation_score(m=10))
    print(y_recommendation)
    
recommendation_score_policy_k(4, MONTE_CARLO_MODEL_PATH)
recommendation_score_policy_k(4, RANDOMIZED_MODEL_PATH)


# mixture_model = MixtureModel(path='dataset', k=K_LAST_PURCHASES, verbose=True, save_path=MONTE_CARLO_MODEL_PATH)
# mixture_model.generate_model(max_iteration=10000)

# # Randomized Model #
# for i in range(K_LAST_PURCHASES):
#     mm = MDP(PATH, k=i+1, save_path=RANDOMIZED_MODEL_PATH)
#     mm.initialise_mdp()
#     mm.randomized_algorithm_for_optimal_policies(N=100)