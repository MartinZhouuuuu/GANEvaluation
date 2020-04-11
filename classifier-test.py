import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from dataset import MultiResolutionDataset, TwoClassDataset
from torchvision import transforms
from classifier_model import FCNET
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn

#80-10-10 split
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

#create loader for combined set
dataloader = DataLoader(combined_set,shuffle = True, batch_size = 8)
# loader = iter(dataloader)

#network
classifier = FCNET(3*2**16, [20], 1)

loss_criterion = nn.BCELoss()
adam_optim = optim.Adam(classifier.parameters())

for epoch in range(2):

	running_loss = 0.0

	for i,batch in enumerate(dataloader):
		images, labels = batch

		adam_optim.zero_grad()

		output = classifier(images)

		loss = loss_criterion(output, labels)
		

		loss.backward()
		adam_optim.step()

		running_loss += loss.item()

		print('iteration', i+1, loss.item())
	print('Epoch',epoch+1,running_loss/250)






