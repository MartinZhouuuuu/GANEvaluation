import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from dataset import MultiResolutionDataset, TwoClassDataset
from torchvision import transforms
from classifier_model import FCNET
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import random_split
from utility import accuracy, imshow, plot_loss, plot_acc

num_epochs = 100
batch_size = 8

transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
		]
	)
#load orignal set
lmdb_set = MultiResolutionDataset('bedroom/00', transform)
lmdb_set.resolution = 256

#load generated set
generated_set = ImageFolder('NVIDIA-bedrooms',transform = transform)

#create combined set
combined_set = TwoClassDataset(lmdb_set,generated_set)

#split train val test
train,val,test = random_split(combined_set, [1600,200,200])
num_train,num_val,num_test = len(train),len(val),len(test)

#create loader for combined set
trainLoader = DataLoader(train,shuffle = True, batch_size = batch_size)
valLoader = DataLoader(val,shuffle = False, batch_size = 200)
testLoader = DataLoader(test,shuffle = False, batch_size = 200)

#network
classifier = FCNET(3*2**16, [20], 1)

loss_criterion = nn.BCELoss()
adam_optim = optim.Adam(classifier.parameters())

train_epoch_loss = []
val_epoch_loss = []
train_epoch_acc = []
val_epoch_acc = []
highest_val_acc = 0
for epoch in range(num_epochs):

	running_loss = 0
	iteration_count = 0
	train_correct = 0

	for i,batch in enumerate(trainLoader):
		iteration_count += 1
		
		train_images, train_labels = batch

		adam_optim.zero_grad()

		train_pred = classifier(train_images).squeeze(1)

		loss = loss_criterion(train_pred, train_labels)
		
		loss.backward()
		adam_optim.step()

		running_loss += loss.item()

		for pred in range(train_labels.size()[0]):
			if train_pred[pred]>=0.5:
				train_pred[pred] = 1
			else:
				train_pred[pred] = 0
			
			if int(train_pred[pred]) == int(train_labels[pred]):
				train_correct += 1

		print('iteration', i+1 ,'train_loss %.3f' % loss.item())

	train_loss = running_loss/iteration_count
	train_acc = train_correct / num_train  * 100
	print('Epoch',epoch+1,'train_loss %.3f' % train_loss,'train_acc', '%.2f'% train_acc + '%')
	
	train_epoch_loss.append(train_loss)
	train_epoch_acc.append(train_acc)

	with torch.no_grad():
		classifier.eval()
		val_images,val_labels = iter(valLoader).next()

		val_pred = classifier(val_images).squeeze(1)
		val_loss = loss_criterion(val_pred, val_labels).item()
		val_acc = accuracy(val_pred, val_labels,num_val)
		if val_acc > highest_val_acc:
			highest_val_acc = val_acc
			torch.save(classifier.state_dict(), 'model-files/%d-%.2f.pth'%(epoch,val_acc))
		print('Epoch', epoch+1, 'val_loss','%.2f'%val_loss,'val_acc', '%.2f'%val_acc + '%')
		
		val_epoch_loss.append(val_loss)
		val_epoch_acc.append(val_acc)

	plot_loss(train_epoch_loss, val_epoch_loss,'plots')
	plot_acc(train_epoch_acc, val_epoch_acc, 'plots')

with torch.no_grad():
	classifier.eval()

	test_images,test_labels = iter(testLoader).next()

	test_pred = classifier(test_images).squeeze(1)

	test_loss = loss_criterion(test_pred, test_labels).item()
	test_acc = accuracy(test_pred, test_labels,num_test)
	print('test_loss','%.2f'%test_loss,'test_acc', '%.2f'%test_acc + '%')
	
'''
print(train_epoch_loss)
print(train_epoch_acc)
print(val_epoch_loss)
print(val_epoch_acc)
'''



