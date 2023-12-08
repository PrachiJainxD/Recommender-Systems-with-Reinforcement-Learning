#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate

#Constants
PATH='dataset'
MONTE_CARLO_MODEL_PATH='saved_models/monte-carlo'
RANDOMIZED_MODEL_PATH='saved_models/randomized-algo'
K_LAST_PURCHASES = 4

## Training Mixture Model with Monte Carlo ##
mixture_model = MixtureModel(path='dataset', k=K_LAST_PURCHASES, verbose=True, save_path=MONTE_CARLO_MODEL_PATH)
mixture_model.generate_model(max_iteration=10000)

## Training Mixture Model with Randomized Algorithm for identifying optimal policies ##
for i in range(K_LAST_PURCHASES):
    mm = MDP(PATH, k=i+1, save_path=RANDOMIZED_MODEL_PATH)
    mm.initialise_mdp()
    mm.randomized_algorithm_for_optimal_policies(N=100)

# ## Perform Hyper-Parameter Tuning
# hyper_parameters_grid_search = {
#     'alpha' : [0.5, 0.75, 1],
#     'k' : 8, #Runs for all MDP until 8
#     'discount' : []
# }
# mixture_model = MixtureModel(path='dataset', k=K_LAST_PURCHASES, verbose=True, save_path=MONTE_CARLO_MODEL_PATH)
# mixture_model.generate_model(max_iteration=10000)

