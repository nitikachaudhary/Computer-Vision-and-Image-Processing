import numpy as np
import cv2
import time

start_time = time.time()


img = cv2.imread('lena_gray.jpg',0)

#Sobel Filters in x and y direction
sobel_x = np.array([[-1, 0, 1], [-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0],[1,2,1]])

#Applying 2D convolution and Normalization
def sobelFilter(img,filter):
    img_height=img.shape[0]
    img_width=img.shape[1]
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
    final_img = np.zeros((img_height, img_width))
    norm_final_img = np.zeros((img_height, img_width))
    for i in range(1,img_width+1):
        for j in range(1,img_height+1):
            neighborhood = img[j-1:j+2, i-1:i+2]        
            final_img[j-1,i-1]=np.sum(np.multiply(neighborhood,filter))
    
    min=np.min(final_img)
    max=np.max(final_img)    
    
    norm_final_img = (final_img-min)/(max-min)             
    
    return norm_final_img


norm_final_img_x = sobelFilter(img,sobel_x)
norm_final_img_y = sobelFilter(img,sobel_y)
print norm_final_img_x

print("--- %s seconds ---" % (time.time() - start_time))
        
cv2.namedWindow('GirlImage',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImage',img)
cv2.namedWindow('GirlImageSober_X',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImageSober_X',norm_final_img_x)
cv2.namedWindow('GirlImageSober_Y',cv2.WINDOW_NORMAL)
cv2.imshow('GirlImageSober_Y',norm_final_img_y)
cv2.waitKey(0)
cv2.destroyAllWindows()