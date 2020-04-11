import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from dataset import MultiResolutionDataset, TwoClassDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	# unnormalize
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show() 

transform = transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
		]
	)

generated_set = ImageFolder('NVIDIA-bedrooms',transform = transform)

lmdb_set = MultiResolutionDataset('00', transform)
lmdb_set.resolution = 256

combined_set = TwoClassDataset(lmdb_set,generated_set)

dataloader = DataLoader(combined_set,shuffle = True, batch_size = 16)
loader = iter(dataloader)
images,labels = next(loader)