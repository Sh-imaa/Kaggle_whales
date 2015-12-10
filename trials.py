import cv2
import numpy as np
from math import pi

img_path = '/media/shimaa/Media/Whales/imgs/whale_34663/w_7001.jpg'
img = cv2.imread(img_path)
h, w = img.shape[:2]
bb_p1 = (340,660)
bb_p2 = (1200,1300)

points = np.array([[42,82],[150,162]])

cv2.rectangle(img, bb_p1, bb_p2, (255,0,0))
img = cv2.resize(img, (w/8,h/8))
rect_center = (bb_p1[0]+(bb_p2[0]-bb_p1[0])/2)/8 , (bb_p1[1]+(bb_p2[1]-bb_p1[1])/2)/8
h1,w1 = (bb_p2[1] - bb_p1[1])/8, (bb_p2[0] - bb_p1[0])/8
M = cv2.getRotationMatrix2D(rect_center, -90, 1.0)
a = np.dot(M,np.array([[42],[82],[1]]))
b = np.dot(M,np.array([[150],[82],[1]]))
c = np.dot(M,np.array([[42],[162],[1]]))
d = np.dot(M,np.array([[150],[162],[1]]))

rotated_img = cv2.warpAffine(img, M, (600,400))
x1 = bb_p1[0]/8
y1 = bb_p1[1]/8
max_x = int(max(a[0], b[0], c[0], d[0]))
min_x = int(min(a[0], b[0], c[0], d[0]))
max_y = int(max(a[1], b[1], c[1], d[1]))
min_y = int(min(a[1], b[1], c[1], d[1]))
rotated_img  = rotated_img[min_y:max_y, min_x:max_x]
img2 = img[y1:y1+h1, x1:x1+w1]
cv2.imshow('rotated', rotated_img);
cv2.imshow('img' , img2); cv2.waitKey(0)