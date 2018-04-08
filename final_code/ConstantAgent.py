import numpy as np


DEFAULT_NUMBER_OF_ASSETS = 10


class ConstantAgent:
    def __init__(self, history_length = 5, number_of_assets = DEFAULT_NUMBER_OF_ASSETS, keeping_proportion=.1):
        """Init a new agent.
        
         """
        self.time = 0
        self.history_length = history_length
        self.history = []
        self.number_of_assets = number_of_assets
        self.keeping_proportion = keeping_proportion

        self.default_action = np.zeros(number_of_assets+1)
        self.default_action[-1] = 1
        
        print('Agent Created')
        pass
    def reset(self):
        self.time = 0
        self.history = []
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        # increasing time
        self.time +=1
        
        action = np.ones(self.number_of_assets + 1)/self.number_of_assets*(1-self.keeping_proportion)
        #action[-1] = 0
        #action = action / np.sum(action[:-1])*(1-self.keeping_proportion)
        action[-1] = self.keeping_proportion
        return action
            
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        # We do nothing during this phase as our agent does not learn anything
        pass
    
