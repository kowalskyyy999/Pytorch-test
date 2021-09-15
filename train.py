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
	train_dataset = utils.MNISTDataset(train=True)

	# Hyper Parameters
	learning_rate = 1e-3
	epochs = 20
	batch_size = 50

	# Load Models ANN
	model = models.MNISTNeuralNetwork()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	metrics = metric.Accuracy_score()

	# Initialization Engine
	engine = utils.Engine(model, optimizer, criterion, metrics, DEVICE)

	# Fitting model of train dataset
	engine.fit(train_dataset, batch_size=batch_size, epochs=epochs, shuffle=True)

	# Visualization of training result
	engine.VisualizationResult()

	# Save the model
	engine.save(path='./model.pth')

