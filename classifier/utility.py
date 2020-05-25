import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def count_correct(pred,labels):
	
	correct = 0
	'''
	class_prediction = pred.clone()
	for i in range(pred.size()[0]):
		if pred[i]>=0.5:
			class_prediction[i] = 1
		else:
			class_prediction[i] = 0
		if int(class_prediction[i]) == int(labels[i]):
			correct += 1
	'''
	#returns the indices of the larger values
	_,predicted = torch.max(pred,1)
	# print(predicted)
	correct += (labels == predicted).sum().item()
	# print(predicted)
	return predicted,correct

def plot_loss(train_loss,val_loss,path):
	plt.plot(list(range(1, len(train_loss)+1)),train_loss)
	plt.plot(list(range(1, len(val_loss)+1)),val_loss)
	plt.title("Loss")
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc = 'lower left')
	plt.savefig(path+'/loss.png')
	plt.close()

def plot_acc(train_acc,val_acc,path):
	plt.plot(list(range(1, len(train_acc)+1)),train_acc)
	plt.plot(list(range(1, len(val_acc)+1)),val_acc)
	plt.title("Accuracy")
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc = 'upper left')
	plt.savefig(path+'/acc.png')
	plt.close()

def matplotlib_imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.cpu().numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(images,predictions,labels,classes):

	# preds, probs = images_to_probs(net, images)
	class_prediction = count_correct(predictions, labels)[0]

	# plot the images in the batch, along with predicted and true labels
	fig = plt.figure(figsize=(10,4))
	
	for idx in np.arange(4):
		ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
		matplotlib_imshow(images[idx])
		ax.set_title("{0}, {1:.3f}\n(label: {2})".format(
			classes[int(class_prediction[idx])],
			predictions[idx],
			classes[int(labels[idx])]),
					color=("green" if int(class_prediction[idx])==int(labels[idx]) else "red"))
	return fig

def predict_and_save(images,predictions,labels,classes,batch_num):
	class_prediction = count_correct(predictions, labels)[0]
	for idx in range(images.size()[0]):
		matplotlib_imshow(images[idx])
		plt.title("{0}, {1}\n(label: {2})".format(
			classes[int(class_prediction[idx])],
			predictions[idx,0].item(),
			classes[int(labels[idx])]),
					color=("green" if int(class_prediction[idx])==int(labels[idx]) else "red"))
		file_name = batch_num + idx
		flag = str(int(class_prediction[idx])==int(labels[idx]))
		plt.savefig('predictions/constant1-vs-2/%d-%.1f-%s.jpg' % (file_name,class_prediction[idx],flag))
		plt.close()
	'''
	for idx in range(images.size()[0]):
		matplotlib_imshow(images[idx])
		plt.title("{0}, {1:.3f}\n(label: {2})".format(
			classes[int(class_prediction[idx])],
			predictions[idx],
			classes[int(labels[idx])]),
					color=("green" if int(class_prediction[idx])==int(labels[idx]) else "red"))
		file_name = batch_num*64 + idx
		flag = str(int(class_prediction[idx])==int(labels[idx]))
		plt.savefig('pred/%s-%.3f-%d.jpg' % (flag,predictions[idx],file_name))
		plt.close()
		'''