#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate

#Constants
PATH='dataset'
MONTECARLO_MODEL_PATH='saved_models/randomized-algo'
HEADERS = ['Rank', 'Game', 'Score']

k = 8
for i in range(k):
    mm = MDP(PATH, k=i+1, save_path=MONTECARLO_MODEL_PATH)
    mm.initialise_mdp()
    mm.randomized_algorithm_for_optimal_policies(N=200)

