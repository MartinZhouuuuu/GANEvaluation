import os
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt

'''
# path = 'images/noise-injection/10/gray'
path = 'images/layer10/gray'
constant_1 = np.empty((256,256))
constant_3 = np.empty((256,256))

for i in range(10000):
	image_id = str(i + 90000) + '.tif'
	# image1 = tifffile.imread(path + '/constant1/' + image_id)
	image1 = tifffile.imread(path + '/modified/' + image_id)
	constant_1 += image1
	# image2 = tifffile.imread(path + '/constant3/' + image_id)
	image2 = tifffile.imread(path + '/unmodified/' + image_id)
	constant_3 += image2
	
overall_diff = constant_1 - constant_3
np.save('0.8_diff.npy',overall_diff)
# np.savetxt('overall_diff.csv', overall_diff, delimiter = ',')

'''
overall_diff = np.load('0.8_diff.npy')

overall_diff = torch.from_numpy(overall_diff)
flattened = overall_diff.view(2**16)
flattened = flattened.numpy()

loaded = torch.load('50-92.55.pth')
weights = loaded['progression.0.weight'].cpu()

class1 = weights[0,:].numpy()
class2 = weights[1,:].numpy()

plt.scatter(flattened,class2,s=5)
plt.xlabel('pixel-diff')
plt.ylabel('weights')
plt.title('class2')
plt.savefig('4.png')


