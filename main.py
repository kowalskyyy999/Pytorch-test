import models
import utils
import metric 

import torch
import torch.nn as nn 
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = utils.MNISTDataset(train = True)
test_dataset = utils.MNISTDataset(train = False)

learning_rate = 1e-3

models = models.MNISTNeuralNetwork()
optimizer = optim.Adam(models.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()
metrics = metric.Accuracy_score()

train_batch_size = 50
epochs = 20

engine = utils.Engine(models, optimizer, criterion, metrics, DEVICE)
engine.fit(train_dataset, batch_size=train_batch_size, epochs = epochs, shuffle=True)
engine.VisualizationResult()
outs, scores, losses = engine.predict(test_dataset, batch_size=50, shuffle=False)

print(outs)
print(scores)
print(losses)
