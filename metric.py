import torch
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class Accuracy_score:
	def __init__(self):
		pass
	
	def __call__(self, true, pred):
		prob = F.softmax(pred, dim=1)
		pred = torch.argmax(prob, dim=1)
		pred = pred.detach().cpu().numpy()
		true = true.detach().cpu().numpy()
		score = accuracy_score(true, pred)
		return score