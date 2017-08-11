import numpy as np
import cv2
import time

start_time = time.time()


img = cv2.imread('lena_gray.jpg',0)

#Separable sobel filters in x and y direction
sobel_x_vert = np.array([[1],[2],[1]])
sobel_x_hori = np.array([[-1,0, 1]])
sobel_y_vert = np.array([[-1],[0],[1]])
sobel_y_hori = np.matrix([[1, 2 ,1]])

img_height = img.shape[0]
img_width = img.shape[1]
   
#Applying vertical 1D convolution
img_x = np.zeros((img_height, img_width))
img_y = np.zeros((img_height, img_width))
img = cv2.copyMakeBorder(img, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
for i in range(1,img_width+1):
    for j in range(1,img_height+1):
        value_x = sobel_x_vert[0,0]*img[j-1,i] + sobel_x_vert[1,0]*img[j,i] + sobel_x_vert[2,0]*img[j+1,i]
        value_y = sobel_y_vert[0,0]*img[j-1,i] + sobel_y_vert[1,0]*img[j,i] + sobel_y_vert[2,0]*img[j+1,i]
        img_x[j-1,i-1]=value_x
        img_y[j-1,i-1]=value_y

#Applying horizontal 1D convolution
img_x = cv2.copyMakeBorder(img_x, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
img_y = cv2.copyMakeBorder(img_y, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
final_img_x = np.zeros((img_height, img_width))
final_img_y = np.zeros((img_height, img_width))
for i in range(1,img_width+1):
    for j in range(1,img_height+1):
        value_x = sobel_x_hori[0,0]*img_x[j,i-1] + sobel_x_hori[0,1]*img_x[j,i] + sobel_x_hori[0,2]*img_x[j,i+1]
        value_y = sobel_y_hori[0,0]*img_y[j,i-1] + sobel_y_hori[0,1]*img_y[j,i] + sobel_y_hori[0,2]*img_y[j,i+1]
        final_img_x[j-1,i-1]=value_x
        final_img_y[j-1,i-1]=value_y


#Normalization for the Gx image
min_x=np.min(final_img_x)
max_x=np.max(final_img_x)    
final_img_x = (final_img_x-min_x)/(max_x-min_x)

#Normalization for the Gy image
min_y=np.min(final_img_y)
max_y=np.max(final_img_y)
final_img_y = (final_img_y-min_y)/(max_y-min_y)


print("--- %s seconds ---" % (time.time() - start_time))
        
cv2.namedWindow('GirlImage',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImage',img)
cv2.namedWindow('GirlImageSober_X',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImageSober_X',final_img_x)
cv2.namedWindow('GirlImageSober_Y',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImageSober_Y',final_img_y)
cv2.waitKey(0)
cv2.destroyAllWindows()