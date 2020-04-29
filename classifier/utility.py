import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
def imshow(img):
	# unnormalize
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show() 

def accuracy(pred,labels,num_samples):
	correct = 0
	for i in range(num_samples):
		if pred[i]>=0.5:
			pred[i] = 1
		else:
			pred[i] = 0
		if int(pred[i]) == int(labels[i]):
			correct += 1
	acc = correct / num_samples * 100 
	return acc

def plot_loss(train_loss,val_loss,path):
	plt.plot(train_loss)
	plt.plot(val_loss)
	plt.title("Loss")
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc = 'upper right')
	plt.savefig(path+'/loss.png')
	plt.close()

def plot_acc(train_acc,val_acc,path):
	plt.plot(train_acc)
	plt.plot(val_acc)
	plt.title("Accuracy")
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc = 'lower right')
	plt.savefig(path+'/acc.png')
	plt.close()