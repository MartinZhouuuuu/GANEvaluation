import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from classifier_model import *
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import random_split
from PIL import Image
from utility import plot_classes_preds,count_correct,predict_and_save
import random
from torchvision.models import resnet50
import tifffile
from PIL import Image
import os

if __name__ == '__main__':
	# device = torch.device('cuda')
	seed = 6489
	torch.manual_seed(seed)
	num_epochs = 300
	batch_size = 64
	dataset_size = 20000
	start_epoch = 0
	
	transform = transforms.Compose(
			[
				transforms.Resize(256,Image.LANCZOS),
				transforms.CenterCrop(256),
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
			]
		)
	
	#create combined set
	combined_set = ImageFolder('real-vs-generated',transform = transform)
	
	





	print(combined_set)
	classes = ['generated','real']
	#for testing all
	loader = DataLoader(
		combined_set,
		shuffle = False,
		batch_size = batch_size,
		num_workers = 8,
		pin_memory=True)

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
		batch_size = batch_size,
		num_workers = 8,
		pin_memory=True)

	testLoader = DataLoader(test,shuffle = False, batch_size = batch_size)

	#network
	classifier = FCNET(3*2**16, [20], 1).to('cuda')
	# classifier = ConvNet([16],include_dense=True).to('cuda')
	# classifier = resnet50(num_classes=1).to('cuda')
	print(classifier)
	# classifier.load_state_dict(torch.load('model-files/100-0.651.pth'))
	
	loss_criterion = nn.BCELoss()
	# loss_criterion = nn.CrossEntropyLoss()
	lr = 1e-4
	adam_optim = optim.Adam(classifier.parameters(), lr=lr)

	# train_epoch_loss = []
	# val_epoch_loss = []
	# train_epoch_acc = []
	# val_epoch_acc = []
	# highest_val_acc = 0
	lowest_val_loss = 999999

	train = True

	if train:
		writer = SummaryWriter('runs/binary_classifier/psi-1.0/constant2/1FC-20')
		writer.add_hparams({'lr': lr,
			'batch_size': batch_size,
			'seed': seed,
			'num_samples': dataset_size
			},{})
		for epoch in range(num_epochs):

			running_loss = 0
			iteration_count = 0
			train_correct = 0

			for i,batch in enumerate(trainLoader):
				iteration_count += 1
				#torch.cuda is really crucial
				train_images, train_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')
				
				if i == 0:
					writer.add_graph(classifier,train_images)
				adam_optim.zero_grad()

				train_pred = classifier(train_images).squeeze(1)

				loss = loss_criterion(train_pred, train_labels)
				
				loss.backward()
				adam_optim.step()

				running_loss += loss.item()
				
				# writer.add_scalar(
					# 'training iteration loss', 
					# loss.item(),
					# global_step=(epoch+1+start_epoch)*len(trainLoader)+i)
				
				train_correct += count_correct(train_pred, train_labels)[1]

				print('iteration', i+1 ,'train_loss %.3f' % loss.item())
				# del train_images,train_labels

			train_loss = running_loss/iteration_count
			train_acc = train_correct / train_size  * 100
			print('Epoch',epoch+1+start_epoch,'train_loss %.3f' % train_loss,'train_acc', '%.2f'% train_acc + '%')
			writer.add_scalar(
				'training epoch loss', 
				train_loss,
				global_step=(epoch+1+start_epoch))

			writer.add_scalar(
				'training epoch acc', 
				train_acc,
				global_step=(epoch+1+start_epoch))

			# train_epoch_loss.append(train_loss)
			# train_epoch_acc.append(train_acc)

			# writer.add_histogram('dense1.weights', classifier.progression[0].weight,epoch+1)

			with torch.no_grad():
				classifier.eval()
				val_running_loss = 0
				val_iteration_count = 0
				val_correct = 0

				random_batch = random.randint(0,31)
				for i,batch in enumerate(valLoader):

					val_images, val_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')
					val_pred = classifier(val_images).squeeze(1)

					val_iteration_loss = loss_criterion(val_pred, val_labels)
					val_running_loss += val_iteration_loss.item()
					if i == random_batch:
						 writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(val_images,val_pred,val_labels,classes),
                            global_step=epoch+1)
					
					val_correct += count_correct(val_pred, val_labels)[1]

					val_iteration_count += 1

				val_loss = val_running_loss / val_iteration_count
				
				val_acc = val_correct / val_size * 100

				writer.add_scalar(
					'val loss', 
					val_loss,
					global_step=(epoch+1+start_epoch))

				writer.add_scalar(
					'val acc', 
					val_acc,
					global_step=(epoch+1+start_epoch))

				if (epoch+1+start_epoch)%10 == 0:
					torch.save(classifier.state_dict(), 'model-files/%d-%.3f.pth'%(epoch+1+start_epoch,val_loss))

				if val_loss <= lowest_val_loss:
					lowest_val_loss = val_loss
					#epochs of model files are wrong
					torch.save(classifier.state_dict(), 'model-files/%d-%.3f.pth'%(epoch+1+start_epoch,val_loss))
					print('new model saved')

				print('Epoch', epoch+1+start_epoch, 'val_loss','%.3f'%val_loss,'val_acc', '%.2f'%val_acc + '%')
				
				# val_epoch_loss.append(val_loss)
				# val_epoch_acc.append(val_acc)

			# plot_loss(train_epoch_loss, val_epoch_loss,'plots')
			# plot_acc(train_epoch_acc, val_epoch_acc, 'plots')

	#should evaluate using best val model
	with torch.no_grad():
		classifier.eval()
		if not train:
			classifier.load_state_dict(torch.load(
				'model-files/real-vs-generated/psi-1.0-nvidia/10k-vs-10k/1Conv1FC-1620/12-97.00.pth'
				# 'model-files/real-vs-generated/psi-1.0-nvidia/10k-vs-10k/1FC-20/52-94.35.pth'
				))
			test_running_loss = 0
			test_iteration_count = 0
			test_correct = 0
			#print scores
			'''
			for file in os.listdir('random/300'):
				image = tifffile.imread('random/300/'+file)
				# print(image)
				image = Image.fromarray(image,'RGB')
				image = transform(image)
				image = image.view(1,3,256,256).cuda()
				# print(image)
				pred = classifier(image).squeeze(1)
				print(pred)
			'''

			for i,batch in enumerate(loader):

				test_images, test_labels = batch[0].to('cuda'), batch[1].type(torch.cuda.FloatTensor).to('cuda')
				test_pred = classifier(test_images).squeeze(1)
				print(test_pred)
				# predict_and_save(test_images, test_pred, test_labels, classes, i)

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
			
			test_acc = test_correct / 10000 * 100
	
			print('test_loss','%.3f'%test_loss,'test_acc', '%.2f'%test_acc + '%')
			
			