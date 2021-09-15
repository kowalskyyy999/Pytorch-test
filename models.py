import torch
import torch.nn as nn 
import torch.nn.functional as F 

class MNISTNeuralNetwork(nn.Module):
	def __init__(self):
		super(MNISTNeuralNetwork, self).__init__()
		self.hidden1 = nn.Linear(784, 100)
		self.hidden2 = nn.Linear(100, 50)
		self.final = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(self.hidden1(x))
		x = F.relu(self.hidden2(x))
		x = self.final(x)
		return x
if __name__ == '__main__':
	pass
	