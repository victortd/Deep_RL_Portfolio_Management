# Read me 
To run the project, you need to put the code in a folder 'final_code' and the data into a folder 'raw_data'.

Steps :
1) open a terminal
2) go to the final code folder 
3) run the command python main.dev with the following arguments
	--agent : default='Agent', Class to use for the agent. Must be in the \'agent\' module.
	--ngames : default='100', number of games to simulate
	--batch : default=1, batch run several agent at the same time
	--hidden_units1 : default='10', number of units in the first hidden layer
	--hidden_units2' : default = '10'number of units in the second hidden layer
	--number_of_assets : default = '10', number of assets to train and run the model
	--buffer_size : default = '10', size of the buffer for the replay buffer
	--batch_size : default = '5', number of samples to train the model
	--gamma : default='.2', reward discount
	--learning_rate : default='.00001', reward discount
	--explo' : default='5', number of exploration games