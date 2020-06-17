import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2

fig = plt.figure(figsize = (14,14))
fig.subplots_adjust(left = 0.01,right = 0.99)
fig.suptitle('Different noise layers',fontsize=50)
fig.tight_layout()

image_name = '90010.tif'

layer_list = [0,1,2,10,12,13]

for i in range(len(layer_list)):
	file_name = 'images/noise-injection/' + str(layer_list[i]) + '/constant1/' + image_name
	# ax = fig.add_subplot(2, 3, i+1)
	image = tifffile.imread(file_name)
	# image = image.astype(np.int16)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	tifffile.imwrite('{}.tif'.format(layer_list[i]),image)

	# ax.imshow(image)
	# ax.axis('off')

# fig.savefig('random.png')
# plt.close(fig)


'''
#comparison
fig = plt.figure(figsize = (8,32))
fig.subplots_adjust(left = 0.01,right = 0.99)
# fig.suptitle('constant vs random',fontsize=50)
fig.tight_layout()

images_list = ['99705','99708','99723','99740','99743','99751','99770','99775','99814']
num_pairs = len(images_list)
for i in range(num_pairs):
	file_name = images_list[i] + '.tif'
	ax = fig.add_subplot(num_pairs, 2, i*2+1)
	image = tifffile.imread('squeeze/rounded/constant/'+ file_name)
	ax.imshow(image)
	ax.axis('off')
	if i == 0:
		ax.set_title(
			'constant',
			fontsize=50
			)

	ax = fig.add_subplot(num_pairs, 2, i*2+2)
	image = tifffile.imread('squeeze/rounded/random/'+ file_name)
	ax.imshow(image)
	if i == 0:
		ax.set_title(
			'random',
			fontsize=50
			)
	
	ax.axis('off')


fig.savefig('1.png')
plt.close(fig)
'''