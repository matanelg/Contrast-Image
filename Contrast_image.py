import cv2
import numpy as np
import matplotlib.pyplot as plt

path_gorilla = "C:/new_computer/Computer Vision/My_Code/4_image_processing/histograms_30_32/gorilla.jpg" # your image path
gorilla = cv2.imread(path_gorilla) # reading image
gorilla_grey = cv2.cvtColor(gorilla,cv2.COLOR_BGR2GRAY)

plt.figure(num=0,figsize=(12,6))
plt.imshow(gorilla_grey,cmap='gray')
plt.title('Gorilla Gray-Scaled Image')

histr = cv2.calcHist([gorilla_grey],[0],None,[256],[0,256])
plt.figure(num=1,figsize=(12,6))
plt.bar(list(range(0,256)),histr[:,0],align='center',color='b')
plt.xlim([0,256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Gray Scale Image Distribution')


eq_gorilla = cv2.equalizeHist(gorilla_grey) # Equalization
histr_eq = cv2.calcHist([eq_gorilla],[0],None,[256],[0,256]) # new distro

plt.figure(num=3,figsize=(12,6))
plt.bar(list(range(0,256)),histr_eq[:,0],align='center',color='b')
plt.plot(histr,color='r')
plt.legend(['Before Equalization','After Equalization'])
plt.xlim([0,256])
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Gray Scale Image Distribution')

plt.figure(num=4,figsize=(12,6))
plt.imshow(eq_gorilla,cmap='gray')






