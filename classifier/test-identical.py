import cv2
import numpy as np
for i in range(100):	
	a = cv2.imread("downloaded/%d.png"%(99999-i))
	b = cv2.imread("adjusted/%d.png"%(99999-i))
difference = cv2.subtract(a, b)    
result = not np.any(difference)
print(result)