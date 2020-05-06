import cv2
import numpy as np
a = cv2.imread("image1.png")
b = cv2.imread("image2.png")
difference = cv2.subtract(a, b)    
result = not np.any(difference)
print(result)