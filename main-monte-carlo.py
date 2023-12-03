#Imports
from mixture_model import MixtureModel
from mdp import MDP
from tabulate import tabulate

#Constants
PATH='dataset'
MONTECARLO_MODEL_PATH='monte-carlo'
HEADERS = ['Rank', 'Game', 'Score']

# Monte Carlo Models with Value Iteration ##
# mm = MDP(PATH, save_path="saved-models-paper")
#mm.initialise_mdp()
# mm.load_policy("mixture-models/mdp-model_k=" + str(3) + ".pkl")
# key = list(mm.policy_list.keys())[0]
# print(key, mm.policy_list[key])

# s = list(mm.T.keys())[0]
# actions = list(mm.T[s].keys())
# for a in actions:
#     for ns in mm.T[s][a]:
#         print(s, a, ns, mm.T[s][a][ns])




# mm = MDP(PATH, save_path=MONTECARLO_MODEL_PATH)
# mm.initialise_mdp()
# v_s = {}
# for s in mm.S:
#     v_s[s] = 0 
# s = list(mm.S.keys())[0]
# v_nxt_i = mm.get_q_value(s, v_s)
# print(mm.A)
# print(v_nxt_i)


mm = MDP(PATH, save_path=MONTECARLO_MODEL_PATH)
mm.initialise_mdp()
# v_s,episode = mm.monte_carlo_value_iteration()
# π_s = mm.get_deterministic_policy(v_s)
# avg_profit = mm.calculate_avg_profit_per_user()

