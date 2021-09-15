import argparse
import models
import utils
import metric 

import torch
import torch.nn as nn 
import torch.optim as optim

if __name__ == '__main__':
	# Initialization device
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load Dataset
	test_dataset = utils.MNISTDataset(train=False)

	# Hyper Parameters
	batch_size = 50
	learning_rate = 1e-3

	# Load Models ANN
	model = models.MNISTNeuralNetwork().to(DEVICE)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	metrics = metric.Accuracy_score()

	# Load ANN Model State Dict
	model.load_state_dict(torch.load('./model.pth'))

	# Initialization Engine
	engine = utils.Engine(model, optimizer, criterion, metrics, DEVICE)

	# Prediction
	pred, score, loss = engine.predict(test_dataset, batch_size=batch_size, shuffle=False)

	print(f'Accuracy Score : {score} -- Loss : {loss}')
