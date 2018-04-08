from __future__ import division
from __future__ import print_function
import numpy as np
import datetime
import os

DEFAULT_NUMBER_OF_ASSETS = 10

# TODO : add a display function
class Environment:
    # List of the possible actions by the agents

    def __init__(self, path_to_data_folder = "../raw_data", initial_cash = 1000000, 
        number_of_assets = DEFAULT_NUMBER_OF_ASSETS, randomized_data = True, debug = False):
        """Instanciate a new environement in its initial state.
        """
        self.path_to_data = path_to_data_folder
        self.initial_cash = initial_cash
        self.number_of_assets = number_of_assets
        self.portfolio = np.zeros(number_of_assets)
        self.debug = debug
        self.relative_gain = 0
        # loading data into the environment
        self.data = np.array(self.load_assets())
        self.sigmas = np.ones(number_of_assets)
        self.scenario_duration = len(self.data[0])
        self.game = -1

        self.printout_matrix = np.ndarray((10, self.number_of_assets+1), dtype = str)


        if randomized_data:
            for i in range(number_of_assets):
                ts = []
                for t in range(self.scenario_duration):
                    ts += [float(self.data[i][t][3])] + [float(self.data[i][t][4])]
                #ts = np.array([float(value[0:][3]) for value in self.data] + [float(value[0:][4]) for value in self.data])
                sigma = np.std(ts)
                self.sigmas[i] = sigma/10
        
        self.reset()
        
    def load_data(self, filename):
        stock_name = filename.split('/')[-1].split('.')[0]
        #keys = ['date', 'coin', 'open', 'high', 'low', 'close', 'volume']

        data = []
        counter = -1        
        with open(filename, 'r') as f:
            for line in f:
                counter +=1

                if counter == 0:
                    continue

                line = line.rstrip().split(",")
                time_stock = datetime.datetime.strptime(line[0][:19], '%d.%m.%Y %H:%M:%S' )

                market_is_open = False
                # making sure the market is open 
                if time_stock.hour<21:
                    if time_stock.hour == 14:
                        if time_stock.minute >=30:
                            market_is_open = True
                    elif time_stock.hour>14:
                        market_is_open = True
                if market_is_open:
                    values = [counter] + [stock_name] +[float(val) for val in line[1:]]
                    data.append(values)


        return data
    
    def normalize_data(self, data):

        s = []
        s = [list(zip(*data))[i] for i in range(2, len(data[0]))]
        j = [np.mean(s[i]) for i in range(0, len(data[0])- 2)]
        std = [np.std(s[i]) for i in range(0, len(data[0])- 2)]
        
        data1 = [(list(zip(*data))[i] -j[i-2])/std[i-2] for i in range(2, len(data[0]))]
        data2 = list(zip(*data))[:2] + data1
        st_data = list(zip(*data2))

        return st_data
    
    def load_assets(self):
        data = []
        filenames = [os.path.join(self.path_to_data, filename) for filename in os.listdir(self.path_to_data)]
        # filenames = np.random.choice(a = filenames, replace = False, size = self.number_of_assets)
        for filename in filenames:
            data.append(self.load_data(filename))
        return data

    def reset(self):
        """Reset the environment for a new run."""
        # place holes
        self.game += 1
        self.remaining_cash = self.initial_cash
        self.time=0
        self.portfolio = np.zeros(self.number_of_assets)
        pass 

    def reset_test(self):
        """Reset the environment for a new run."""
        # place holes
        self.game += 1
        self.remaining_cash = self.initial_cash
        self.time += 1
        self.portfolio = np.zeros(self.number_of_assets)
        pass 
    
    def observe(self, time=None, value = 'open'):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        values = {'open': 2, 'close': 5}
        if time == None:
            time = self.time
        state_of_the_market = np.array([float(asset[time,values[value]])+np.random.normal(loc = 0, scale = self.sigmas[index]) for index, asset in enumerate(self.data)] +
            [self.remaining_cash], dtype=np.float)
        observation = state_of_the_market
        return observation
    
    def compute_portfolio_value(self):
        investments = self.portfolio
        values = np.array([market_values for market_values in  self.observe(self.time, 'close')[:-1]])
        return np.sum(investments*values)
    
    def compute_portfolio(self, action, time = None, value = 'open'):
        """function to get the number of action the agent has at a given time"""
        
        action = action.reshape(self.number_of_assets+1)
        cash_investment = self.remaining_cash * action[:-1]
        number_of_stocks = cash_investment/self.observe(value=value)[:-1]
        return number_of_stocks
    
    def act(self, action):
        action_ = action.reshape(self.number_of_assets+1)

        print('time : ', self.time)
        print('game : ', self.game)
        print('investments : ', end = "\t")
        for value in action_ : 
            print(round(value*100, 1), ' %', end = '\t')
        print()

        if self.debug :print("action : ", action)
        
        
        # computing the current portfolio value at time t
        previous_portfolio_value = self.compute_portfolio_value()
        
        # updating the portfolio
        self.portfolio = self.compute_portfolio(action, value = 'close')
        # print(self.portfolio)
        # computing the value of the new portfolio at time t
        current_porfolio_value_t = self.compute_portfolio_value()
        # updating the remaining money
        self.remaining_cash += previous_portfolio_value - current_porfolio_value_t
        
        # checking if the remaining money is still greater than 0
        if self.debug : print("previous_portfolio_value ",previous_portfolio_value)
        if self.debug : print("current_porfolio_value_t ", current_porfolio_value_t)
        if self.debug : print("rem_money", self.remaining_cash)
        gain = self.remaining_cash + current_porfolio_value_t
        relative_gain = round((gain-self.initial_cash)/self.initial_cash*100, 1)
        print("gain : ", gain)
        print("relative gain : ", "+ " if relative_gain>0 else "- ", np.abs(relative_gain) , " %")
        print("remaining cash : ", self.remaining_cash)
        self.relative_gain = relative_gain
        
        if self.remaining_cash < 0 :
            
            return (- self.scenario_duration/max(self.time,.1), "Agent is broke")
        
        # incrementing time
        self.time += 1
        
        # if the agent has lived through the whole scenario,
        # return the total return
        if self.time >= self.scenario_duration:
            reward = (current_porfolio_value_t+self.remaining_cash -self.initial_cash)/self.initial_cash
            print('reward : ', reward)
            return (reward, "Trading completed")
                    #/self.initial_cash
        # else, we compute the value of the portfolio at time t + 1
        current_portfolio_value_t_plus_1 = self.compute_portfolio_value()
        reward = (current_portfolio_value_t_plus_1-current_porfolio_value_t)
        reward = relative_gain#*self.initial_cash
        print('reward : ', reward)
        print("-"*150)

        return (reward, None)
                #/(current_porfolio_value_t+0.00001)*self.initial_cash
    
    def display(self):
        print('Remaining money : {}'.format(self.remaining_cash))
        print('Current porfolio value : {}'.format(self.compute_portfolio_value()))
