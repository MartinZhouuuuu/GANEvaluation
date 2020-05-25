import torch
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2
from classifier_model import FCNET

net = FCNET(3*2**16, [], 2)
with torch.no_grad():	
	net.eval()
	net.load_state_dict(torch.load(
		'record/model-files/binary-classifier/constant1-vs-2/no-hidden-unit/19-0.080.pth'))
	loaded = torch.load(
		'record/model-files/binary-classifier/constant1-vs-2/no-hidden-unit/19-0.080.pth')
	weights = loaded['progression.0.weight']
	square = weights.view(2,3,256,256)
	constant1 = np.transpose(square[0,:,:,:].cpu().squeeze().numpy(),(1,2,0))
	constant2 = np.transpose(square[1,:,:,:].cpu().squeeze().numpy(),(1,2,0))
	tifffile.imsave('constant1.tif',constant1)
	tifffile.imsave('constant2.tif',constant2)

	constant1_r = constant1[:,:,0]
	constant1_g = constant1[:,:,1]
	constant1_b = constant1[:,:,2]

	constant2_r = constant2[:,:,0]
	constant2_g = constant2[:,:,1]
	constant2_b = constant2[:,:,2]

	tifffile.imsave('constant1_r.tif',constant1_r)
	tifffile.imsave('constant1_g.tif',constant1_g)
	tifffile.imsave('constant1_b.tif',constant1_b)
	tifffile.imsave('constant2_r.tif',constant2_r)
	tifffile.imsave('constant2_g.tif',constant2_g)
	tifffile.imsave('constant2_b.tif',constant2_b)