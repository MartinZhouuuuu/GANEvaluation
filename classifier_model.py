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


class ConvNet(nn.Module):

	def __init__(self,conv_config,include_dense = False):
		super().__init__()
		'''
		conv_config: a list of conv layer filter numbers
		include_dense: whether to include a classification layer
		'''
		self.num_conv = len(conv_config)
		self.conv_progression = nn.ModuleList([])
		self.conv_progression.append(nn.Conv2d(3, conv_config[0], 3,padding=1))
		self.conv_progression.append(nn.ReLU())
		self.conv_progression.append(nn.MaxPool2d(2))
		

		for i in range(self.num_conv-1):
			self.conv_progression.append(nn.Conv2d(conv_config[i],conv_config[i+1],3,padding=1))
			self.conv_progression.append(nn.ReLU())
			self.conv_progression.append(nn.MaxPool2d(2))
			
		self.linear_progression = nn.ModuleList([])
		
		self.conv_output = 2**((8-self.num_conv)*2)*conv_config[-1]

		if include_dense:
			self.linear_progression.append(nn.Linear(self.conv_output,20))
			self.linear_progression.append(nn.ReLU())
			self.linear_progression.append(nn.Linear(20,1))
		else:
			self.linear_progression.append(nn.Linear(self.conv_output,1))
		self.linear_progression.append(nn.Sigmoid())

	def forward(self,x):
		for i in range(len(self.conv_progression)):	
			x = self.conv_progression[i](x)
		x = x.view(-1,self.conv_output)
		
		for i in range(len(self.linear_progression)):
			x = self.linear_progression[i](x)
		
		return x

	def __len__(self):
		return len(self.conv_progression) + len(self.linear_progression)
