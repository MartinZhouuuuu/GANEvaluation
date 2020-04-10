import torch
from model import StyledGenerator
from torchvision import utils 

num_samples = 10

generator = StyledGenerator().cuda()
ckpt = torch.load('checkpoint/425000.model')
generator.load_state_dict(ckpt['generator'])
generator.eval()

with torch.no_grad():
	for i in range(num_samples):	
		image = generator(torch.randn(1,512),step = 6,alpha = 1).data().cuda()
		utils.save_image(images, 'generated/%d.png'%i)






