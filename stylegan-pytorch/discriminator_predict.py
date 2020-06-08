import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Discriminator
from torchvision.datasets import ImageFolder
from dataset import MultiResolutionDataset
from torchvision import transforms
from torch.nn import functional as F
#image in (-1,1)

device = torch.device('cuda')
discriminator = Discriminator(from_rgb_activate= False).cuda()

ckpt = torch.load('checkpoint/500000.model')
discriminator.load_state_dict(ckpt['discriminator'])
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #convert to (-1,1)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
discriminator.eval()


#noise
noise = torch.randn(10,3,256,256).cuda()
noise_scores = discriminator(noise,step = 6,alpha = 1)
print(noise_scores)

#generated images


#NVIDIA images
nvidia_images = ImageFolder('NVIDIA-bedrooms',transform=transform)
nvidia_loader = DataLoader(
	nvidia_images,shuffle=False, num_workers= 0, batch_size= 10
	)
loader = iter(nvidia_loader)

with torch.no_grad():
	all_images = next(loader)[0].cuda()
	nvidia_scores = discriminator(all_images, step = 6, alpha = 2)



#Original images
#get a new set
# dataset = MultiResolutionDataset('', transform)
