import torch
import torch.nn.functional as F
import numpy 
import idx2numpy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class MNISTDataset(Dataset):
	def __init__(self, root = './data', train = True):
		self.root = root
		self.imagesArray, self.labelsArray = self.preprocessing(train)
		
	def __len__(self):
		return len(self.imagesArray)
	
	def __getitem__(self, idx):
		imageArray = self.imagesArray[idx]
		labelArray = self.labelsArray[idx]
		
		image_tensor = torch.from_numpy(imageArray).float()
		label_tensor = torch.tensor(labelArray).long()
		
		image_tensor = image_tensor.view(-1)
		
		return image_tensor, label_tensor
	
	def preprocessing(self, train):
		if train:
			imagesFile = 'train-images-idx3-ubyte'
			labelsFile = 'train-labels-idx1-ubyte'
		else:
			imagesFile = 't10k-images-idx3-ubyte'
			labelsFile = 't10k-labels-idx1-ubyte'
			
		imagesArray = idx2numpy.convert_from_file(self.root + '/' + imagesFile)
		labelsArray = idx2numpy.convert_from_file(self.root + '/' + labelsFile)
		
		return imagesArray, labelsArray

class Engine:
	def __init__(self, model, optimizer, criterion, metrics, device):
		self.model = model.to(device)
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.train_Acc = []
		self.train_Loss = []
		self.metrics = metrics
		
	def fit(self, dataset, batch_size = 50, epochs = 20, num_workers = 0, pin_memory = False, shuffle=False):

		dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle, num_workers = num_workers, pin_memory = pin_memory)
				
		for epoch in range(epochs):
			scores = 0
			losses = 0

			self.model.train()
			tk = tqdm(dataloader, total = len(dataloader))
			for images, labels in tk:
				images = images.to(device=self.device)
				labels = labels.to(device=self.device)
				
				self.optimizer.zero_grad()
				
				out = self.model(images)
				
				loss = self.criterion(out, labels)
				
				loss.backward()
				
				self.optimizer.step()
				
				score = self.metrics(labels, out)
				
				losses += loss.item()*images.size(0)
				scores += score * images.size(0) * 100
				
				tk.set_postfix({'Epoch': epoch + 1, 'Loss': loss.item(), 'Accuracy': score})
				
			self.train_Loss.append(losses/len(dataloader.dataset))
			self.train_Acc.append(scores/len(dataloader.dataset))
			
	def predict(self, dataset, batch_size = 50, num_workers = 0, pin_memory = False, shuffle=False):
		scores = 0
		losses = 0
		outs = []

		dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)

		with torch.no_grad():
			self.model.eval()
			tk = tqdm(dataloader, total=len(dataloader))
			for images, labels in tk:
				images = images.to(device = self.device)
				labels = labels.to(device = self.device)
				
				out = self.model(images)
				loss = self.criterion(out, labels)
				
				score = self.metrics(labels, out)
				
				losses += loss.item()*images.size(0)
				scores += score*images.size(0)*100
				
				tk.set_postfix({'Loss': loss.item(), 'Accuracy': score})

				outs.append(self._pred_processing(out))
				
		losses = losses/len(dataloader.dataset)
		scores = scores/len(dataloader.dataset)
		
		return outs, scores, losses

	@staticmethod
	def _pred_processing(out):
		prob = F.softmax(out, dim= -1)
		pred = torch.argmax(prob, dim= -1)
		return pred

	def single_predict(self, image):
		image_tensor = torch.tensor(image).float()
		image_tensor = image_tensor.view(-1).unsqueeze(0)
		out = self.model(image_tensor.to(device = self.device))
		out = out.squeeze(0)
		pred = self._pred_processing(out)
		return pred.detach().cpu().numpy()


	def save(self, path):
		torch.save(self.model.state_dict(), path)

	def VisualizationResult(self, name = './result.png'):
		fig, ax = plt.subplots(1, 2, figsize=(16, 8))
		fig.suptitle('Training Plot MNIST Classification', fontsize = 20)

		ax[0].plot(self.train_Acc)
		ax[0].set_title('Plot Accuracy Score', fontsize= 14)
		ax[0].set_ylabel('Accuracy')

		ax[1].plot(self.train_Loss)
		ax[1].set_title('Plot Losses', fontsize=14)
		ax[1].set_ylabel('Loss')

		plt.savefig(name)
		plt.show()
		