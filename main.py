#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate

#Constants
PATH='dataset'
MIXTURE_MODEL_PATH='mixture-models'
HEADERS = ['Rank', 'Game', 'Score']

## Markov Chain Mixture Models with Policy Iteration ##

#Training Mixture Model
# mixture_model = MixtureModel(path='dataset', k=3, verbose=False, save_path='saved_models/'+MIXTURE_MODEL_PATH)
# mixture_model.generate_model(max_iteration=1)
# results  = mixture_model.predict(187131847)
# print(tabulate([[ind+1, r[0], r[1]] for ind, r in enumerate(results)], HEADERS, "psql"))

#Training Mixture Model
mixture_model = MixtureModel(path='dataset', k=8, verbose=True, save_path='saved_models/'+MIXTURE_MODEL_PATH)
mixture_model.generate_model(max_iteration=10000)
#results  = mixture_model.predict(187131847)
#print(tabulate([[ind+1, r[0], r[1]] for ind, r in enumerate(results)], HEADERS, "psql"))

#Comparing against a pre-trained model
# mixture_model = MixtureModel(path='dataset', k=3, verbose=False, save_path='saved-models-paper/mixture-models')
# results= mixture_model.predict(187131847)
# print(tabulate([[ind+1, r[0], r[1]] for ind, r in enumerate(results)], HEADERS, "psql"))