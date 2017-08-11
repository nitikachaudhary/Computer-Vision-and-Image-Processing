import numpy as np
import cv2


#Reading the image and its dimensions
img = cv2.imread('lena_gray.jpg',0)
img_height=img.shape[0]
img_width=img.shape[1]

#Sobel Filters in x and y direction
sobel_x = np.array([[-1, 0, 1], [-2,0,2],[-1,0,1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0],[1,2,1]])

#Separable Sobel Filters in x and y direction
sobel_x_vert = np.array([[1],[2],[1]])
sobel_x_hori = np.array([[-1,0, 1]])
sobel_y_vert = np.array([[-1],[0],[1]])
sobel_y_hori = np.matrix([[1, 2 ,1]])

#Function: Adding an extra one pixel border by replicating border pixels
def padding(img):
    image = np.zeros((img_height, img_width))
    image = cv2.copyMakeBorder(img, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
    return image
    
#Creating two intial padded images for 1D and 2D
img_1D=padding(img)
img_2D=padding(img)

#Initializing the final G=Gx2+Gy2
G = np.zeros((img_height, img_width))

#Applying 2D convolution
final_img_x = np.zeros((img_height, img_width))
final_img_y = np.zeros((img_height, img_width))
for i in range(1,img_width+1):
    for j in range(1,img_height+1):
        neighborhood = img_2D[j-1:j+2, i-1:i+2]        
        value_x=np.sum(np.multiply(neighborhood,sobel_x))
        value_y=np.sum(np.multiply(neighborhood,sobel_y))
        
        final_img_x[j-1,i-1]=value_x
        final_img_y[j-1,i-1]=value_y
        
        #Gradient Magnitude 
        G[j-1,i-1]=np.sqrt(value_x**2 + value_y**2)

def normalize(img):
    minimum=np.min(img)
    maximum=np.max(img)    
    
    img = (img-minimum)/(maximum-minimum)
    return img


#Final normalized images after 2D convolution with sobel filters
img_x_2D = normalize(final_img_x)
img_y_2D = normalize(final_img_y)
G = normalize(G)

#Applying vertical 1D convolution
img_x = np.zeros((img_height, img_width))
img_y = np.zeros((img_height, img_width))
for i in range(1,img_width+1):
    for j in range(1,img_height+1):
        value_x = sobel_x_vert[0,0]*img_1D[j-1,i] + sobel_x_vert[1,0]*img_1D[j,i] + sobel_x_vert[2,0]*img_1D[j+1,i]
        value_y = sobel_y_vert[0,0]*img_1D[j-1,i] + sobel_y_vert[1,0]*img_1D[j,i] + sobel_y_vert[2,0]*img_1D[j+1,i]
        img_x[j-1,i-1]=value_x
        img_y[j-1,i-1]=value_y

#Applying horizontal 1D convolution
img_x = cv2.copyMakeBorder(img_x, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
img_y = cv2.copyMakeBorder(img_y, 1, 1, 1, 1,cv2.BORDER_REPLICATE)
img_x_1D = np.zeros((img_height, img_width))
img_y_1D = np.zeros((img_height, img_width))
for i in range(1,img_width+1):
    for j in range(1,img_height+1):
        value_x = sobel_x_hori[0,0]*img_x[j,i-1] + sobel_x_hori[0,1]*img_x[j,i] + sobel_x_hori[0,2]*img_x[j,i+1]
        value_y = sobel_y_hori[0,0]*img_y[j,i-1] + sobel_y_hori[0,1]*img_y[j,i] + sobel_y_hori[0,2]*img_y[j,i+1]
        img_x_1D[j-1,i-1]=value_x
        img_y_1D[j-1,i-1]=value_y


#Final normalized images after 1D convolution with sobel filters
img_x_1D=normalize(img_x_1D)
img_y_1D=normalize(img_y_1D)
 
#Verifying the results after 1D and 2D convolution were the same       
print "img x 1D == img x 2D" ,np.all(img_x_1D-img_x_2D==0)
print "img y 1D == img y 2D", np.all(img_y_1D-img_y_2D==0)

cv2.namedWindow('Image_G',cv2.WINDOW_NORMAL)
cv2.imshow('Image_G',G)
cv2.namedWindow('Image_2D_X',cv2.WINDOW_NORMAL)
cv2.imshow('Image_2D_X',img_x_2D)
cv2.namedWindow('Image_2D_Y',cv2.WINDOW_NORMAL)
cv2.imshow('Image_2D_Y',img_y_2D)
cv2.namedWindow('Image_2D_Y',cv2.WINDOW_NORMAL)
cv2.imshow('Image_1D_X',img_x_1D)
cv2.namedWindow('Image_1D_Y',cv2.WINDOW_NORMAL)
cv2.imshow('Image_1D_Y',img_y_1D)

cv2.waitKey(0)
cv2.destroyAllWindows()