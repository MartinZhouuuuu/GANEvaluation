import torch
from torch import nn
from torch.nn import functional as F

class FCNET(nn.Module):
	'''
	num_units is a list of number of hidden units 
	'''
	def __init__(self,input_dim,num_units,out_dim):
		super().__init__()
		self.input_dim = input_dim
		self.num_hidden_layer = len(num_units)
		self.progression = nn.ModuleList([])
		self.progression.append(nn.Linear(self.input_dim,num_units[0]))
		self.progression.append(nn.ReLU())

		for i in range(self.num_hidden_layer-1):
			self.progression.append(nn.Linear(num_units[i],num_units[i+1]))
			self.progression.append(nn.ReLU())

		self.progression.append(nn.Linear(num_units[-1],out_dim))
		self.progression.append(nn.Sigmoid())

	def forward(self,x):
		x = x.view(-1,self.input_dim)
		for i in range(len(self)):	
			x = self.progression[i](x)
		
		return x

	def __len__(self):
		return len(self.progression) 
