import cv2
import glob
import numpy as np
import os
from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel

img = glob.glob('./stereo/centre/*.png')
img.sort()
print(img)
data=[]
final = []
fx, fy, cx, cy, G_camera_image, LUT=ReadCameraModel("./model")
for i in range(len(img)):
	temp = cv2.imread(img[i],0)
	data.append(temp) 
count = 0
EXTN = ".png"
for j in range(len(data)):
	count = count + 1
	# if(count>50):
	# break
	pic = data[j]
	print(count)
	tempo = cv2.cvtColor(pic, cv2.COLOR_BayerGR2BGR)
	temp_img = UndistortImage(tempo, LUT)
	filename = str(count).zfill(4) + EXTN
	cv2.imwrite("./undistorted_input_images/frame%s" % filename, temp_img) 
	# print(temp_img)
	final.append(temp_img)
cv2.imshow("image",final[0])
cv2.waitKey(0)
cv2.destroyAllWindows()



