import tifffile
import cv2
import os
path = 'images/constant1-vs-2/constant2'
for filename in os.listdir(path):
	image = tifffile.imread(path+'/'+filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	tifffile.imsave('gray/'+filename,gray)