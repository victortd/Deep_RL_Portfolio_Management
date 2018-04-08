import environment
import runner
import agent

import argparse

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment', metavar='ENV_CLASS', type=str, default='Environment', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--ngames', type=int, metavar='n', default='100', help='number of games to simulate')
parser.add_argument('--batch', type=int, metavar='nagent', default=1, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', help='Display cumulative results at each step')
parser.add_argument('--hidden_units1', type = int, metavar = 'n_hidden_units1', default='10', help= 'number of units in the first hidden layer')
parser.add_argument('--hidden_units2', type = int, metavar = 'n_hidden_units2', default = '10', help = 'number of units in the second hidden layer')
parser.add_argument('--number_of_assets', type = int, metavar='n_assets', default = '10', help='number of assets to train and run the model')
parser.add_argument('--buffer_size', type = int, metavar='buf_size', default = '10', help='size of the buffer for the replay buffer')
parser.add_argument('--batch_size', type=int, metavar='training_size', default = '5', help='number of samples to train the model')
parser.add_argument('--gamma', type=float, metavar='Gamma', default='.2', help='reward discount')
parser.add_argument('--learning_rate', type=float, metavar='lr', default='.00001', help='reward discount')
parser.add_argument('--explo', type=int, metavar='explo', default='5', help='number of exploration games')


def main():
    args = parser.parse_args()
    agent_class = eval('agent.{}'.format(args.agent))
    env_class = eval('environment.{}'.format(args.environment))

    print("Running a batched simulation with {} agents in parallel...".format(args.batch))
    my_runner = runner.BatchRunner(env_maker = env_class, agent_maker = agent_class, count = args.batch,
    hidden_units1 = args.hidden_units1, hidden_units2=args.hidden_units2, number_of_assets = 10, 
    buffer_size=args.buffer_size, batch_size=args.batch_size, learning_rate=args.learning_rate, 
    GAMMA = args.gamma, exploration_game = args.explo, verbose =False)
    final_reward = my_runner.loop(args.ngames, None)
    print("Obtained a final average reward of {}".format(final_reward))


if __name__ == "__main__":
    main()