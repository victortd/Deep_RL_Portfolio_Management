import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Lambda, add
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

DEFAULT_NUMBER_OF_ASSETS = 10

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, 
                 LEARNING_RATE=0.001, HIDDEN1_UNITS = 2, HIDDEN2_UNITS = 2):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.HIDDEN1_UNITS = HIDDEN1_UNITS
        self.HIDDEN2_UNITS = HIDDEN2_UNITS

        #K.set_session(sess)

        # creating the actor network
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        # creating the model network
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        # creating a placeholder for the gradient
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        # creating a gradient for the weights
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        # creating the optimization process 
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        # initializing the parameters
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        # running the learning
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
        pass

    def target_train(self):
        # getting the weights of the actor network
        actor_weights = self.model.get_weights()
        # getting the weights from the actor_target network
        actor_target_weights = self.target_model.get_weights()
        # weighted average of the weights 
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        # updating the weights of the target_actor network
        self.target_model.set_weights(actor_target_weights)
        pass

    def create_actor_network(self, state_size,action_dim):
        # defining the input layer (given by the number of parameters needed to fully describe a state)
        # in our case it is going to be market values and remaining money
        S = Input(shape=[state_size])   
        # creating the first dense layer
        h0 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        # creating the second dense layer
        h1 = Dense(self.HIDDEN2_UNITS, activation='relu')(h0)
        # creating the outputs
        V = Dense(action_dim, activation = 'softmax')(h1) 
        # joining all together
        model = Model(inputs=S,outputs=V)
        # returning the model, its weights and the input layer
        return model, model.trainable_weights, S

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU,
                 LEARNING_RATE= 0.01, HIDDEN1_UNITS=2, HIDDEN2_UNITS=2):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.HIDDEN1_UNITS = HIDDEN1_UNITS
        self.HIDDEN2_UNITS = HIDDEN2_UNITS
        
        # K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_0state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        # creating the network layers for the state
        S = Input(shape=[state_size]) 
        w1 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(self.HIDDEN2_UNITS, activation='linear')(w1)
        # creating the network layers for the action
        A = Input(shape=[action_dim],name='action2')   
        a1 = Dense(self.HIDDEN2_UNITS, activation='linear')(A) 
        # merging the networks
        h2 = add([h1,a1])
        # adding another layer
        h3 = Dense(self.HIDDEN2_UNITS, activation='relu')(h2)
        # creating output layer
        V = Dense(action_dim,activation='linear')(h3)
        # joining the input and the output
        model = Model(inputs=[S,A],outputs=V)
        # optimizer definition
        adam = Adam(lr=self.LEARNING_RATE)
        # compilation of the model
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 

# for the replay buffer, we are going to store the Nth latest state, the action taken, the reward and the new state 
# the idea being to create smaller batch to train the model

from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class DDPGAgent:
    def __init__(self, hidden_units1 = 2, hidden_units2 = 2, 
                 number_of_assets = DEFAULT_NUMBER_OF_ASSETS,
                 buffer_size = 100, batch_size = 30,
                 learning_rate = 0.1, GAMMA = .2,
                 exploration_game = 5, session = tf.Session()):
        """Init a new agent.
        
         """
        self.sess = session
        self.exploration_game = exploration_game
        self.hidden_units1 = hidden_units1
        self.hidden_units2 = hidden_units2
        self.number_of_assets = number_of_assets
        self.learning_rate = learning_rate
        self.time = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        # K.set_session(self.sess)

        self.actor_network = ActorNetwork(sess=self.sess, 
                                          state_size=number_of_assets+1,
                                          action_size = number_of_assets+1,
                                          BATCH_SIZE=batch_size, TAU = .5, 
                                          LEARNING_RATE = learning_rate, 
                                          HIDDEN1_UNITS= hidden_units1, 
                                          HIDDEN2_UNITS=hidden_units2)
        self.critic_network = CriticNetwork(sess=self.sess, 
                                            action_size = number_of_assets+1,
                                            state_size=number_of_assets+1, 
                                            BATCH_SIZE=batch_size, TAU = .5, 
                                            LEARNING_RATE=learning_rate, 
                                            HIDDEN1_UNITS = hidden_units1, 
                                            HIDDEN2_UNITS=hidden_units2)
        
        self.initialized = False
        self.initial_money = None
        self.remaining_money = None
        self.initial_portfolio = None
        self.previous_portfolio = None
        self.current_portfolio = None
        self.previous_state = None
        self.current_state = None
        self.GAMMA = GAMMA
        self.total_reward = 0
        self.game = 0
        self.previous_action = np.zeros(number_of_assets)
        
        
        
        print('Agent Created')
        pass
    def reset(self):
        self.time = 0
        self.game += 1
        self.total_reward = 0
        self.remaining_money = self.initial_money
        self.current_portfolio = self.initial_portfolio
        self.previous_action = np.zeros(self.number_of_assets+1)
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)
      
        pass

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        
        self.time +=1
        
        if not self.initialized:
            self.initial_money = observation[1]
            self.remaining_money = self.initial_money
            self.initialized = True
        
        self.current_portfolio = observation[:-1]
        self.remaining_money = observation[-1]
        self.current_state = observation
        
        if self.game < self.exploration_game:# or np.random.uniform()<.1:
            # returning no investment actions
            null_action = np.random.uniform(size=(self.number_of_assets+1)) * np.random.binomial(size = (self.number_of_assets+1), p = .5, n = 1) + 0.000001
            null_action = null_action / np.sum(null_action)#[:-1])/10
            #null_action[-1] = .9

            return null_action
        else:
            # predicting the action based on the actor network
            action = self.actor_network.model.predict(self.current_state.reshape(1, self.current_state.shape[0]))
            return action
            
    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        loss = 0
        # print('action ', action)
        # print(self.actor_network.model.get_weights()[-2])
        # we need to skip the first step to get a previous step
        if self.time > 1:
            # state, action, reward, new_state, done
            self.replay_buffer.add(self.current_state, self.previous_action, 
                                   self.previous_state, self.previous_reward, None)
            # if the size of the buffer has exceeded the buffersize we can create a minibatch and start learning
            if self.time > self.batch_size + 1:
                # print("-"*180 +"> "+str(self.time))
                # learning time
                batch = self.replay_buffer.getBatch(self.batch_size)
                states = np.zeros((self.batch_size, self.number_of_assets+1))
                for i in range(self.batch_size):
                    states[i,:] = batch[i][2]
                actions = np.zeros((self.batch_size, self.number_of_assets+1))
                for i in range(self.batch_size):
                    actions[i,:] = batch[i][1]
 
                # states = np.array([e[2] for e in batch])

                #actions = np.array([e[1] for e in batch])
                rewards = np.array([e[3] for e in batch])
                new_states = np.array([e[0] for e in batch])
                # dones = np.array([e[4] for e in batch], ndmin= 2).reshape(1, self.batch_size).T
                y_t = actions
                # predicting the target Q values
                target_q_values = self.critic_network.target_model.predict([new_states, self.actor_network.target_model.predict(new_states)])  
                # weighting the values 
                y_t = self.GAMMA * target_q_values + rewards.reshape(self.batch_size, 1)
                #print(states)
                loss += self.critic_network.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actions#self.actor_network.model.predict(states)
                grads = self.critic_network.gradients(states, a_for_grad)
                # print('a_for_grad', a_for_grad)
                # print('grads', grads)
                # print("loss", loss)
                self.actor_network.train(states, grads)
                self.actor_network.target_train()
                self.critic_network.target_train()
                
        
                
                
        # keeping the previous values
        self.previous_state = self.current_state
        self.previous_reward = reward
        self.previous_action = action        
        self.total_reward += reward
        pass
    
