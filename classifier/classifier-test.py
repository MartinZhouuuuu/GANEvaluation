import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from classifier_model import *
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import random_split
from utility import accuracy, imshow, plot_loss, plot_acc
from PIL import Image

if __name__ == '__main__':
	# device = torch.device('cuda')
	torch.manual_seed(6489)
	num_epochs = 100
	batch_size = 64
	dataset_size = 20000

	transform = transforms.Compose(
			[
				transforms.Resize(256,Image.LANCZOS),
				transforms.CenterCrop(256),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
			]
		)
	
	#create combined set
	combined_set = ImageFolder('original-vs-generated',transform = transform)
	print(combined_set)

	#split train val test
	train_size = int(dataset_size*0.8)
	val_size = int((dataset_size - train_size)/2)
	test_size = val_size
	train,val,test = random_split(combined_set, [train_size,val_size,test_size])

	#create loader for combined set
	trainLoader = DataLoader(
		train,
		shuffle = True, 
		batch_size = batch_size,
		num_workers = 8,
		pin_memory=True)

	valLoader = DataLoader(
		val,
		shuffle = False, 
		batch_size = 64,
		num_workers = 8,
		pin_memory=True)

	testLoader = DataLoader(test,shuffle = False, batch_size = 64)

	#network
	# classifier = FCNET(3*2**16, [20,20,20,20,20,20,20], 1).to('cuda')
	classifier = ConvNet([16,16,16,16],include_dense=True).to('cuda')
	print(classifier)
	loss_criterion = nn.BCELoss()
	adam_optim = optim.Adam(classifier.parameters(), lr=0.001)

	train_epoch_loss = []
	val_epoch_loss = []
	train_epoch_acc = []
	val_epoch_acc = []
	highest_val_acc = 0

	train = True
	
	if train:
		for epoch in range(num_epochs):

			running_loss = 0
			iteration_count = 0
			train_correct = 0

			for i,batch in enumerate(trainLoader):
				iteration_count += 1
				#torch.cuda is really crucial
				train_images, train_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')

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
				del train_images,train_labels

			train_loss = running_loss/iteration_count
			train_acc = train_correct / train_size  * 100
			print('Epoch',epoch+1,'train_loss %.3f' % train_loss,'train_acc', '%.2f'% train_acc + '%')
			
			train_epoch_loss.append(train_loss)
			train_epoch_acc.append(train_acc)

			with torch.no_grad():
				classifier.eval()
				val_running_loss = 0
				val_iteration_count = 0
				val_correct = 0

				for i,batch in enumerate(valLoader):

					val_images, val_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')
					
					val_pred = classifier(val_images).squeeze(1)

					val_iteration_loss = loss_criterion(val_pred, val_labels)
					val_running_loss += val_iteration_loss.item()

					for index in range(val_labels.size()[0]):
						if val_pred[index]>=0.5:
							val_pred[index] = 1
						else:
							val_pred[index] = 0
						
						if int(val_pred[index]) == int(val_labels[index]):
							val_correct += 1

					val_iteration_count += 1

				val_loss = val_running_loss / val_iteration_count
				
				val_acc = val_correct / val_size * 100

				if val_acc > highest_val_acc:
					highest_val_acc = val_acc
					#epochs of model files are wrong
					torch.save(classifier.state_dict(), 'model-files/%d-%.2f.pth'%(epoch+1,val_acc))
					print('new model saved')
				
				print('Epoch', epoch+1, 'val_loss','%.2f'%val_loss,'val_acc', '%.2f'%val_acc + '%')
				
				val_epoch_loss.append(val_loss)
				val_epoch_acc.append(val_acc)

			plot_loss(train_epoch_loss, val_epoch_loss,'plots')
			plot_acc(train_epoch_acc, val_epoch_acc, 'plots')

	#should evaluate using best val model
	with torch.no_grad():
		classifier.eval()
		if not train:
			classifier.load_state_dict(torch.load('model-files/28-93.65.pth'))
			test_running_loss = 0
			test_iteration_count = 0
			test_correct = 0

			for i,batch in enumerate(testLoader):

				test_images, test_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')
				
				test_pred = classifier(test_images).squeeze(1)

				test_iteration_loss = loss_criterion(test_pred, test_labels)
				test_running_loss += test_iteration_loss.item()

				for index in range(test_labels.size()[0]):
					if test_pred[index]>=0.5:
						test_pred[index] = 1
					else:
						test_pred[index] = 0
					
					if int(test_pred[index]) == int(test_labels[index]):
						test_correct += 1

				test_iteration_count += 1

			test_loss = test_running_loss / test_iteration_count
			
			test_acc = test_correct / test_size * 100
	
			print('test_loss','%.2f'%test_loss,'test_acc', '%.2f'%test_acc + '%')
