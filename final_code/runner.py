from __future__ import division
from __future__ import print_function
import numpy as np
import datetime
import os
import tensorflow as tf
import keras.backend as K

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """
    def __init__(self, env_maker, agent_maker, count, 
        hidden_units1 = 10, hidden_units2 = 10, 
        number_of_assets = 10, buffer_size = 30, 
        batch_size = 5, learning_rate = .1, 
        GAMMA = .2, 
        exploration_game = 5, verbose=False):

        self.session = tf.Session()
        K.set_session(self.session)

        if str(agent_maker)== "<class 'DDPG.DDPGAgent'>":
            def agent_maker_session():
                return agent_maker(hidden_units1 = hidden_units1, hidden_units2 = hidden_units2, 
                     number_of_assets = number_of_assets,
                     buffer_size = buffer_size, batch_size = batch_size,
                     learning_rate = learning_rate, GAMMA = GAMMA,
                     exploration_game = exploration_game,
                     session = self.session)

            self.agents = iter_or_loopcall(agent_maker_session, count)
        else : 
            self.agents = iter_or_loopcall(agent_maker, count)
        self.environments = iter_or_loopcall(env_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]


    def loop(self, games, max_iter = None):

        testing_reward = 0
        if max_iter is None:
            max_iter = self.environments[0].scenario_duration
        for (agent, env) in zip(self.agents, self.environments):
            training_length = int(max_iter*.6)
            for g in range(1, games+1):
                agent.reset()
                env.reset()
                for i in range(1, training_length):
                    observation = env.observe()
                    action = agent.act(observation)
                    (reward, stop) = env.act(action)
                    agent.reward(observation, action, reward)
                    if stop is not None :
                        break
            env.reset_test()
            agent.reset()
            for i in range(training_length, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                if stop is not None:
                    break
            testing_reward += env.relative_gain
        return testing_reward/games


    