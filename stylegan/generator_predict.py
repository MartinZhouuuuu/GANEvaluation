import torch
from model import StyledGenerator
from torchvision import utils 

device = torch.device('cpu')
num_samples = 10

generator = StyledGenerator()
ckpt = torch.load('checkpoint/305000.model',map_location=device)
generator.load_state_dict(ckpt['generator'])
generator.eval()

with torch.no_grad():
	for i in range(num_samples):	
		image = generator(torch.randn(1,512),step = 6,alpha = 1)
		utils.save_image(image, 
			'generated/%d.png'%i, 
			range = (-1,1),
			normalize = True)






