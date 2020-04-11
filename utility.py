import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
	# unnormalize
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show() 