import numpy as np


DEFAULT_NUMBER_OF_ASSETS = 10


class MeanReverseAgent:
    def __init__(self, history_length = 5, number_of_assets = DEFAULT_NUMBER_OF_ASSETS, keeping_proportion = .9):
        """Init a new agent.
        
         """
        self.time = 0
        self.history_length = history_length
        self.history = []

        self.default_action = np.zeros(number_of_assets+1)
        self.default_action[-1] = 1
        self.keeping_proportion = keeping_proportion
        self.number_of_assets = number_of_assets
        
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
        # taking action
        if len(self.history) < self.history_length:
        	self.history.append(observation[:-1])
        	return self.default_action
        else :
            # investing everything on the least performing asset
        	index = np.argmin(np.array(self.history[-1])- np.array(self.history[0]))
        	action = np.zeros(self.number_of_assets+1)
            action[index]= 1 - self.keeping_proportion
            action[-1] = self.keeping_proportion
            self.history = self.history[1:] + [observation[:-1]]
        	return action
            
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        # We do nothing during this phase as our agent does not learn anything
        pass
    
