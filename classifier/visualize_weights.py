import torch
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2
from classifier_model import FCNET

loaded = torch.load(
	'record/model-files/13-0.090.pth')
weights = loaded['progression.0.weight']
class1 = weights[0,:]
class2 = weights[1,:]

square1 = class1.view(1,256,256)
square2 = class2.view(1,256,256)
constant1 = np.transpose(square1.cpu().numpy(),(1,2,0))
constant2 = np.transpose(square2.cpu().numpy(),(1,2,0))

square = weights.view(2,1,256,256)
constant3 = np.transpose(square[0,:,:,:].cpu().numpy(),(1,2,0))
constant4 = np.transpose(square[1,:,:,:].cpu().numpy(),(1,2,0))

tifffile.imsave('constant1.tif',constant1)
tifffile.imsave('constant2.tif',constant2)
tifffile.imsave('constant3.tif',constant3)
tifffile.imsave('constant4.tif',constant4)


'''
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
'''