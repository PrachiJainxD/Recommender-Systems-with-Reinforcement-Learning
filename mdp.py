import operator
import pickle
import os
from tabulate import tabulate
import numpy as np
import math
import pandas as pd

from mdp_handler import MDPInitializer


class MDP:
    """
    Class to run the MDP.
    """

    def __init__(self, path='data', alpha=1, k=3, discount_factor=0.999, verbose=True, save_path="saved_models"):
        """
        The constructor for the MDP class.
        :param path: path to data
        :param alpha: the proportionality constant when considering transitions
        :param k: the number of items in each state
        :param discount_factor: the discount factor for the MDP
        :param verbose: flag to show steps
        :param save_path: the path to which models should be saved and loaded from
        """

        # Initialize the MDPInitializer
        self.mdp_i = MDPInitializer(path, k, alpha)
        self.df = discount_factor
        self.verbose = verbose
        self.save_path = save_path
        # The set of states
        self.S = {}
        # The set of state values
        self.V = {}
        # The set of actions
        self.A = []
        # The set of transitions
        self.T = {}
        # The policy of the MDP
        self.policy = {}
        # A policy list
        self.policy_list = {}
        #Per Iteration Diff Metric
        self.iteration_vs_reward = []

    def print_progress(self, message):
        if self.verbose:
            print(message)

    def initialise_mdp(self):
        """
        The method to initialise the MDP.
        :return: None
        """

        # Initialising the actions
        self.print_progress("Getting set of actions.")
        self.A = self.mdp_i.actions
        self.print_progress("Set of actions obtained.")

        # Initialising the states, state values, policy
        self.print_progress("Getting states, state-values, policy.")
        self.S, self.V, self.policy, self.policy_list = self.mdp_i.generate_initial_states()
        self.print_progress("States, state-values, policy obtained.")

        # Initialise the transition table
        self.print_progress("Getting transition table.")
        self.T = self.mdp_i.generate_transitions(self.S, self.A)
        self.print_progress("Transition table obtained.")

    #Markov Chain Mixture Models
    
    def one_step_lookahead(self, state):
        """
        Helper function to calculate state-value function.
        :param state: state to consider
        :return: action values for that state
        """

        # Initialise the action values and set to 0
        action_values = {}
        for action in self.A:
            action_values[action] = 0

        # Calculate the action values for each action
        for action in self.A:
            for next_state, P_and_R in self.T[state][action].items():
                if next_state not in self.V:
                    self.V[next_state] = 0
                # action_value +=  probability * (reward + (discount * next_state_value))
                action_values[action] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[next_state]))

        return action_values

    def update_policy(self):
        """
        Helper function to update the policy based on the value function.
        :return: None
        """
        new_reward = 0
        for state in self.S:
            action_values = self.one_step_lookahead(state)
            # The action with the highest action value is chosen
            self.policy[state] = max(action_values.items(), key=operator.itemgetter(1))[0]
            new_reward += action_values[self.policy[state]]
            self.policy_list[state] = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)
        return new_reward / len(self.S)

    def policy_eval(self):
        """
        Helper function to evaluate a policy
        :return: estimated value of each state following the policy and state-value
        """

        # Initialise the policy values
        policy_value = {}
        for state in self.policy:
            policy_value[state] = 0

        # Find the policy value for each state and its respective action dictated by the policy
        for state, action in self.policy.items():
            for next_state, P_and_R in self.T[state][action].items():
                if next_state not in self.V:
                    self.V[next_state] = 0
                # policy_value +=  probability * (reward + (discount * next_state_value))
                policy_value[state] += P_and_R[0] * (P_and_R[1] + (self.df * self.V[next_state]))

        return policy_value

    def no_change_in_policy(self, policy_prev):
        """
        Helper function to compare the given policy with the current policy
        :param policy_prev: the policy to compare with
        :return: a boolean indicating if the policies are different or not
        """
        total_diff = 0
        for state in policy_prev:
            # If the policy does not match even once then return False
            if policy_prev[state] != self.policy[state]:
                return False
        return True

    def policy_iteration(self, max_iteration=1000, start_where_left_off=False, to_save=True):
        """
        Algorithm to solve the MDP
        :param max_iteration: maximum number of iterations to run.
        :param start_where_left_off: flag to load a previous model(set False if not and filename otherwise)
        :param to_save: flag to save the current model
        :return: None
        """

        # Load a previous model
        if start_where_left_off:
            self.load(start_where_left_off)

        # Start the policy iteration
        policy_prev = self.policy.copy()
        for i in range(max_iteration):
            self.print_progress("Iteration" + str(i) + ":")

            # Evaluate given policy
            self.V = self.policy_eval()
            
            # Improve policy
            new_reward = self.update_policy()
            self.iteration_vs_reward.append(new_reward)

            if self.no_change_in_policy(policy_prev):
                self.print_progress("Policy converged at iteration " + str(i+1))
                break
            policy_prev = self.policy.copy()
        
        print('Reward vs Iteration = ', self.iteration_vs_reward)
        # Save the model
        if to_save:
            self.save("mdp-model_k=" + str(self.mdp_i.k) + ".pkl")

    def save(self, filename):
        """
        Method to save the trained model
        :param filename: the filename it should be saved as
        :return: None
        """

        self.print_progress("Saving model to " + filename)
        os.makedirs(self.save_path, exist_ok=True)
        with open(self.save_path + "/" + filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Method to load a previous trained model
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        self.print_progress("Loading model from " + filename)
        try:
            with open(self.save_path + "/" + filename, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
        except Exception as e:
            print(e)

    def save_policy(self, filename):
        """
        Method to save the policy
        :param filename: the filename it should be saved as
        :return: None
        """

        self.print_progress("Saving model to " + filename)
        os.makedirs(self.save_path, exist_ok=True)
        with open(self.save_path + "/" + filename, 'wb') as f:
            pickle.dump(self.policy_list, f, pickle.HIGHEST_PROTOCOL)

    def load_policy(self, filename):
        """
        Method to load a previous policy
        :param filename: the filename from which the model should be extracted
        :return: None
        """

        self.print_progress("Loading model from " + filename)
        try:
            with open(self.save_path + "/" + filename, 'rb') as f:
                self.policy_list = pickle.load(f)
        except Exception as e:
            print(e)

    def recommend(self, user_id):
        """
        Method to provide recommendation to the user
        :param user_id: the user_id of a given user
        :return: the game that is recommended
        """
        user_id = str(user_id)
        # self.print_progress("Recommending for " + str(user_id))
        pre = []
        for i in range(self.mdp_i.k - 1):
            pre.append(None)
        try:
            games = pre + self.mdp_i.transactions[user_id]
        except:
            print('User Not Found. Try from Below User List - ')
            print(self.mdp_i.transactions.keys())

        # for g in games[self.mdp_i.k-1:]:
        #     print(self.mdp_i.games[g], self.mdp_i.game_price[g])

        user_state = ()
        for i in range(len(games) - self.mdp_i.k, len(games)):
            user_state = user_state + (games[i],)
        # print(self.mdp_i.game_price[self.policy[user_state]])
        # return self.mdp_i.games[self.policy[user_state]]

        rec_list = []
        # print(self.policy_list)
        #print(user_state)

        if 'policy_list' in self.policy_list:
            self.policy_list = self.policy_list['policy_list']
        
        for game_details in self.policy_list[user_state]:
            rec_list.append((self.mdp_i.games[game_details[0]], game_details[1]))

        return rec_list

    def evaluate_decay_score(self, alpha=10):
        """
        Method to evaluate the given MDP using exponential decay score
        :param alpha: a parameter in exponential decay score
        :return: the average score
        """

        transactions = self.mdp_i.transactions.copy()

        user_count = 0
        total_score = 0
        # Generating a testing for each test case
        for user in transactions:
            total_list = len(transactions[user])
            if total_list == 1:
                continue

            score = 0
            for i in range(1, total_list):
                self.mdp_i.transactions[user] = transactions[user][:i]

                rec_list = self.recommend(user)
                rec_list = [rec[0] for rec in rec_list]
                m = rec_list.index(self.mdp_i.games[transactions[user][i]]) + 1
                score += 2 ** ((1 - m) / (alpha - 1))

            score /= (total_list - 1)
            total_score += 100 * score
            user_count += 1

        return total_score / user_count

    def evaluate_recommendation_score(self, m=10):
        """
        Function to evaluate the given MDP using exponential decay score
        :param m: a parameter in recommendation score score
        :return: the average score
        """

        transactions = self.mdp_i.transactions.copy()

        user_count = 0
        total_score = 0
        # Generating a testing for each test case
        for user in transactions:
            temp = self.mdp_i.transactions[user].copy()

            total_list = len(transactions[user])
            if total_list == 1:
                continue

            item_count = 0
            for i in range(1, total_list):
                self.mdp_i.transactions[user] = transactions[user][:i]

                rec_list = self.recommend(user)
                rec_list = [rec[0] for rec in rec_list]
                rank = rec_list.index(self.mdp_i.games[transactions[user][i]]) + 1
                if rank <= m:
                    item_count += 1

            score = item_count / (total_list - 1)
            total_score += 100 * score
            user_count += 1
            self.mdp_i.transactions[user] = temp 

        return total_score / user_count

    def evaluate_recommendation_score2(self, k=10):
        """
        Function to evaluate the given MDP using exponential decay score
        :param m: a parameter in recommendation score score
        :return: the average score
        """
        return 0
        # transactions = self.mdp_i.transactions.copy()

        # user_count = 0
        # total_score = 0
        # # Generating a testing for each test case
        # for user in transactions:
        #     temp = self.mdp_i.transactions[user].copy()
        #     rec_list = self.recommend(user)
        #     print(rec_list[0][0])
        #     total_score += rec_list[0][0][1]
        #     user_count += 1

        # return total_score / user_count
    
    def recommend_best_action(self, user_id, π_s):
        """
        Method to provide recommendation to the user
        :param user_id: the user_id of a given user
        :return: the game that is recommended
        """
        user_id = str(user_id)
        # self.print_progress("Recommending for " + str(user_id))
        pre = []
        for i in range(self.mdp_i.k - 1):
            pre.append(None)
        try:
            games = pre + self.mdp_i.transactions[user_id]
        except:
            print('User Not Found. Try from Below User List - ')
            print(self.mdp_i.transactions.keys())
        # for g in games[self.mdp_i.k-1:]:
        #     print(self.mdp_i.games[g], self.mdp_i.game_price[g])

        user_state = ()
        for i in range(len(games) - self.mdp_i.k, len(games)):
            user_state = user_state + (games[i],)
        # print(self.mdp_i.game_price[self.policy[user_state]])
        # return self.mdp_i.games[self.policy[user_state]]

        rec_list = []
        # print(self.policy_list)
        #print(user_state)
        #print(user_state)
        return π_s[user_state]

    def calculate_avg_profit(self, π_s):
        transactions = self.mdp_i.transactions.copy()
        user_count = 0
        total_score = 0
        # Generating a testing for each test case
        avg_profit = []
        for user in transactions:
            total_list = len(transactions[user])
            # if user == '151603712':
            #     print(total_list)
            # if total_list == 1:
            #     continue
            item_count = 0
            avg_profit_per_user = []
            total_profit = 0
            for i in range(1, total_list):
                best_possible_action = self.recommend_best_action(user, π_s)
                total_profit += self.mdp_i.games[best_possible_action][1]
                avg_profit_per_user.append(total_profit)
            avg_profit.append(np.average(avg_profit_per_user))
        return np.average(avg_profit)
    
    
    #Monte Carlo Mixture Model

    def init_v_s(self):
        v_s = {}
        for s in self.S:
            v_s[s] = 0
        return v_s

    def init_policy(self):
        π_s = {}
        for s in self.S:
            π_s[s] = None
        return  π_s       

    def get_q_value(self, s, v_s,γ=0.9):
        q_nxt_i = []
        print_stmts = False
        if print_stmts:
            print('Initial State - ', s)
        for action_i, a in enumerate(self.A):
            q_a = 0.0
            if print_stmts:
                print('\n')
            for next_state in self.T[s][a]:               
                p, R_s = self.T[s][a][next_state]
                if next_state in v_s:
                    q_a += p * (R_s + γ*v_s[next_state])
                    if print_stmts:
                        print(a, p, R_s, q_a, v_s[next_state])
            q_nxt_i.append(q_a)
            if print_stmts:
                print(q_nxt_i)
        return q_nxt_i
    
    def monte_carlo_value_iteration(self,γ=0.9):
        episode = 0
        v_s = self.init_v_s()
        while True:
            v_s_updated = self.init_v_s()
            episode += 1
            Δ = float('-inf')
            for s in self.S:
                v_s_i = v_s[s]
                q_s_a = self.get_q_value(s, v_s)
                v_s_updated[s] = np.max(q_s_a)
                Δ = max(Δ, abs(v_s_updated[s]- v_s_i))
            print('Episode No - ', episode, 'with Δ = ', Δ)
            v_s = v_s_updated.copy()
            # if episode == 1:
            #     return v_s,episode
            π_s = self.get_deterministic_policy(v_s)
            avg_profit = self.calculate_avg_profit(π_s)
            print(avg_profit)
            if Δ < 0.001 and episode > 1:
                return v_s,episode

    def get_deterministic_policy(self, v_s):
        π_s = self.init_policy()
        for s in self.S:
            q_s_a = self.get_q_value(s, v_s)
            action_idx = np.argmax(q_s_a)
            π_s[s] = self.A[action_idx]
        return π_s

    



