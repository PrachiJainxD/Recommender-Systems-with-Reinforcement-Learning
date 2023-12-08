import operator
import pickle
import os
from tabulate import tabulate
import numpy as np
import math
import pandas as pd
import random

from mdp_handler import MDPInitializer


class MDP:
    """
    Class to run the MDP.
    """

    def __init__(self, path='data', alpha=1, beta_weight=1, k=3, discount_factor=0.999, verbose=True, save_path="saved_models"):
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
        self.mdp_i = MDPInitializer(path, k, alpha, beta_weight)
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

    def step(self, state, action):
        """
        Take an action in the current state and return the next state and reward.
        :param state: The current state
        :param action: The action to be taken
        :return: A tuple of the next state and the reward for taking the action
        """

        # Check if the state is in the transition table, handle if not
        if state not in self.T:
            # Handle the missing state
            # For example, you could return the same state with zero reward
            # or initialize the missing state in the transition table
            return state, 0

        # Check if the action is valid for the current state
        if action not in self.T[state]:
            raise ValueError(f"Action '{action}' is not valid for state {state}")

        # Get the possible transitions for the given state and action
        possible_transitions = self.T[state][action]

        # Select the next state based on the transition probabilities
        # We will use a random choice weighted by the probabilities
        next_states, probabilities_rewards = zip(*possible_transitions.items())
        probabilities = [pr[0] for pr in probabilities_rewards]
        next_state = random.choices(next_states, weights=probabilities, k=1)[0]

        # Get the reward for the transition to the next state
        reward = possible_transitions[next_state][1]
        return next_state, reward


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
            new_reward = sum(self.V.values())
            self.iteration_vs_reward.append(new_reward)

            if self.no_change_in_policy(policy_prev):
                self.print_progress("Policy converged at iteration " + str(i+1))
                break
            policy_prev = self.policy.copy()
        
        print('Reward vs Iteration = ', self.iteration_vs_reward)
        # Save the model``
        if to_save:
            self.save("mdp-model_k=" + str(self.mdp_i.k) + ".pkl")

    def is_value_stable(self, value_prev, threshold=0.01):
        """
        Helper function to check if the value function has stabilized.
        :param value_prev: the previous value function to compare with
        :param threshold: the threshold for considering a change as significant
        :return: a boolean indicating if the value function is stable or not
        """
        for state in value_prev:
            if abs(value_prev[state] - self.V[state]) > threshold:
                return False
        return True

    def calc_reward(self):
        new_reward = 0
        for state in self.S:
            action_values = self.one_step_lookahead(state)
            new_reward += action_values[self.policy[state]]
        return new_reward / len(self.S)

    def sarsa_algorithm_for_optimal_policies(self, N=1000, alpha=1.0, gamma=0.9, epsilon=0.5, stability_threshold=0.05, to_save=True):
        best_policy = None
        best_policy_performance = float('-inf')
        value_prev = self.V.copy()  # Copy of the initial state values

        # Initialize Q-values
        Q = {state: {action: 0 for action in self.A} for state in self.S}

        for i in range(N):
            print('Episode - ', i + 1)

            # Initialize state
            state = random.choice(list(self.S))

            # Choose action from state using policy derived from Q (ε-greedy)
            action = self.choose_action(state, Q, epsilon)
            diff = []
            while True:
                # Take action and observe reward and next state
                next_state, reward = self.step(state, action)

                # Choose next action from next state using policy derived from Q (ε-greedy)
                next_action = self.choose_action(next_state, Q, epsilon)
                diff.append(alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action]))
                # SARSA Update
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

                state, action = next_state, next_action

                # Derive policy from Q-values
                for s in self.S:
                    self.policy[s] = max(Q[s], key=Q[s].get)

                # Evaluate policy performance
                self.V = self.policy_eval()
                performance = sum(self.V.values())
                self.iteration_vs_reward.append(performance)

                if performance > best_policy_performance:
                    best_policy_performance = performance
                    best_policy = self.policy.copy()

                # Check if the value function has stabilized
                if self.is_value_stable(value_prev, stability_threshold):
                    print('Value function stabilized at episode - ', i + 1)
                    break

                # Update the previous value function for the next iteration
                value_prev = self.V.copy()
            print(np.mean(diff))
            print('Best Policy Performance - ', best_policy_performance)
            self.policy = best_policy

            if to_save:
                self.save("sarsa_mdp-model_k=" + str(self.mdp_i.k) + ".pkl")

    def choose_action(self, state, Q, epsilon):
        # Check if the state is in the Q-table, add if not
        if state not in Q:
            Q[state] = {action: 0 for action in self.A}

        if random.uniform(0, 1) < epsilon:
            # Exploration: choose a random action
            return random.choice(self.A)
        else:
            # Exploitation: choose the best action based on state values
            # Calculate the value of each action by considering the state value of the resulting state
            return max(Q[state], key=Q[state].get)

    def is_value_stable(self, value_prev, threshold):
        """
        Check if the value function has stabilized.
        :param value_prev: the previous value function
        :param threshold: the threshold for considering a change as significant
        :return: Boolean indicating if the value function is stable
        """
        for state in value_prev:
            if abs(value_prev[state] - self.V[state]) > threshold:
                return False
        return True

    def q_learning_for_optimal_policies(self, N=100, alpha=1.0, gamma=0.9, epsilon=0.5, max_steps_per_episode=10, to_save=True):
        best_policy = None
        best_policy_performance = float('-inf')

        # Initialize Q-values (action values)
        Q = {state: {action: 0 for action in self.A} for state in self.S}

        for i in range(N):
            print('Episode - ', i + 1)

            # Initialize state
            state = random.choice(list(self.S))

            for step in range(max_steps_per_episode):
                # Choose action from state using policy derived from Q (ε-greedy)
                action = self.choose_Q_action(state, Q, epsilon)

                # Take action and observe reward and next state
                next_state, reward = self.step(state, action)

                # Check if the next state is in the Q-table, add if not
                if next_state not in Q:
                    Q[next_state] = {action: 0 for action in self.A}

                # Q-learning Update
                max_next_q = max(Q[next_state].values())
                Q[state][action] += alpha * (reward + gamma * max_next_q - Q[state][action])

                state = next_state

            # Derive policy from Q-values
            for s in self.S:
                self.policy[s] = max(Q[s], key=Q[s].get)

            # Evaluate policy performance
            self.V = self.policy_eval()
            performance = sum(self.V.values())
            self.iteration_vs_reward.append(performance)

            if performance > best_policy_performance:
                best_policy_performance = performance
                best_policy = self.policy.copy()

        print('Best Policy Performance - ', best_policy_performance)
        self.policy = best_policy

        if to_save:
            self.save("q_learning_mdp-model_k=" + str(self.mdp_i.k) + ".pkl")


    def choose_Q_action(self, state, Q, epsilon):
        """
        Choose an action based on an ε-greedy policy derived from Q-values.
        :param state: The current state
        :param Q: The dictionary of Q-values
        :param epsilon: The probability of choosing a random action (exploration)
        :return: The chosen action
        """
        if random.uniform(0, 1) < epsilon:
            # Exploration: choose a random action
            return random.choice(self.A)
        else:
            # Exploitation: choose the best action based on Q-values
            return max(Q[state], key=Q[state].get)


    def td_learning_for_optimal_policies(self, N=100, alpha=0.7, gamma=0.7, epsilon=0.5, max_steps_per_episode=10, to_save=True):
        best_policy = None
        best_policy_performance = float('-inf')

        # Initialize V-values (state values)
        V = {state: 0 for state in self.S}

        for i in range(N):
            print('Episode - ', i + 1)

            # Initialize state
            state = random.choice(list(self.S))

            for step in range(max_steps_per_episode):
                # Choose action from state using policy derived from V (ε-greedy)
                action = self.choose_TD_action(state, V, epsilon)

                # Take action and observe reward and next state
                next_state, reward = self.step(state, action)

                # Check if the next state is in the value function, add if not
                if next_state not in V:
                    V[next_state] = 0  # Initialize with a default value, e.g., 0

                # TD Update
                V[state] += alpha * (reward + gamma * V[next_state] - V[state])

                state = next_state

            # Derive policy from V-values
            for s in self.S:
                self.policy[s] = self.choose_TD_action(s, V, 0)  # 0 epsilon for greedy policy

            # Evaluate policy performance
            self.V = self.policy_eval()
            performance = sum(self.V.values())
            self.iteration_vs_reward.append(performance)

            if performance > best_policy_performance:
                best_policy_performance = performance
                best_policy = self.policy.copy()

        print('Best Policy Performance - ', best_policy_performance)
        self.policy = best_policy

        if to_save:
            self.save("td_mdp-model_k=" + str(self.mdp_i.k) + ".pkl")

    def choose_TD_action(self, state, V, epsilon):
        """
        Choose an action based on an ε-greedy policy derived from state values.
        :param state: The current state
        :param V: The dictionary of state values
        :param epsilon: The probability of choosing a random action (exploration)
        :return: The chosen action
        """
        if random.uniform(0, 1) < epsilon:
            # Exploration: choose a random action
            return random.choice(self.A)
        else:
            # Exploitation: choose the best action based on state values
            action_values = {}
            for action in self.A:
                # Assume step method returns the next state and reward for a given state and action
                next_state, _ = self.step(state, action)

                # Check if the next state is in the value function, add if not
                if next_state not in V:
                    V[next_state] = 0  # Initialize with a default value, e.g., 0

                action_values[action] = V[next_state]

            # Choose the action with the highest value
            return max(action_values, key=action_values.get)


    
    def randomized_algorithm_for_optimal_policies(self, N=5, to_save=True):
        best_polcicy = None
        best_policy_performance = 0
        n_actions = len(self.mdp_i.actions)-1
        
        for i in range(N):
            print('Iteration i - ', i+1)
            for s in self.S:
                self.policy[s] = self.mdp_i.actions[random.randint(0, n_actions)]
            self.V = self.policy_eval()
            performance = sum(self.V.values())
            #performance = self.calc_reward()
            self.iteration_vs_reward.append(performance)
            if performance > best_policy_performance:
                best_policy_performance = performance
                best_policy = self.policy.copy()
        print('Best Policy Performance - ', best_policy_performance)
        self.policy = best_policy
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


