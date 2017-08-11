import cv2
import numpy as np
import sys

#read image as a grayscale image
left_img = cv2.imread('view1.png', 0)
right_img = cv2.imread('view5.png', 0)

#reading ground truth
gt_left = cv2.imread('disp1.png', 0)
gt_right = cv2.imread('disp5.png', 0)


height = left_img.shape[0]
width = left_img.shape[1]

#Disparity without consistency check
disparity_L = np.zeros((height,width),dtype=np.uint8)
disparity_R = np.zeros((height,width),dtype=np.uint8)

#Disparity with consistency check
disparity_L_cons = np.zeros((height,width),dtype=np.uint8)
disparity_R_cons = np.zeros((height,width),dtype=np.uint8)

#threshold to match stereo image pair, if the disparity goes beyond this its rejected
threshold=75

#size of kernel(pad =1 for 3x3, 4 for 9x9)
pad=4

#Padding the image
img_L = cv2.copyMakeBorder(left_img, pad, pad, pad, pad,cv2.BORDER_CONSTANT,0)
img_R = cv2.copyMakeBorder(right_img, pad, pad, pad, pad,cv2.BORDER_CONSTANT,0)



# Calculating left disparity map(disparity estimate matching the left image to right image)
for k in range (pad,height+pad):
    for i in range(pad,width+pad):
        neighborhood_L = img_L[k-pad:k+pad+1, i-pad:i+pad+1]
        min_mse= sys.maxint
        disparity=0
        j=i
        while  j > pad-1:                        
            neighborhood_R = img_R[k-pad:k+pad+1, j-pad:j+pad+1]
            mse = np.sum((neighborhood_L - neighborhood_R) ** 2)
            if mse < min_mse:
                min_mse=mse
                disparity=i-j
            j=j-1;
        disparity_L[k-pad,i-pad]=disparity

#Calculating right disparity map
for k in range (pad,height+pad):
    for i in range(pad,width+pad):
        neighborhood_R = img_R[k-pad:k+pad+1, i-pad:i+pad+1]
        min_mse= sys.maxint
        disparity=0
        j=i;
        while  j < width+pad:               
            neighborhood_L = img_L[k-pad:k+pad+1, j-pad:j+pad+1]
            mse = np.sum((neighborhood_R - neighborhood_L) ** 2)
            if mse < min_mse:
                min_mse=mse
                disparity=j-i
            j=j+1;
        disparity_R[k-pad,i-pad]=disparity

#Calculating MSE for left and right disparity with respect to ground truth
mse_Left = np.mean((disparity_L - gt_left) ** 2)
mse_Right = np.mean((disparity_R- gt_right) ** 2)

print "mse Left" ,mse_Left
print "mse Right", mse_Right

#Consistency check for left disparity
def consistencyCheckLeft():
    for i in range (0,left_img.shape[0]):
        for j in range (0,left_img.shape[1]):
            xr=j-disparity_L[i,j]
            if xr<width and xr>=0:
                xl=xr+disparity_R[i,xr]
                if j==xl:
                    disparity_L_cons[i,j]=disparity_L[i,j] 

#consistency check for right disparity    
def consistencyCheckRight():
    for i in range (0,right_img.shape[0]):
        for j in range (0,right_img.shape[1]):
            xl=j+disparity_R[i,j]
            if xl<width and xl>=0:
                xr=xl-disparity_L[i,xl]
                if j==xr:
                    disparity_R_cons[i,j]=disparity_R[i,j] 

def consGroundTruth(disparity,gt):
    for i in range(0,disparity.shape[0]):
        for j in range(0,disparity.shape[1]):
            if disparity[i,j] ==0:
                gt[i,j]=0    

consistencyCheckLeft() 
consistencyCheckRight()
consGroundTruth(disparity_L_cons,gt_left)
consGroundTruth(disparity_R_cons,gt_right)


                
#Calculating MSE
mse_Left_cons = np.mean((disparity_L_cons - gt_left) ** 2)
mse_Right_cons = np.mean((disparity_R_cons - gt_right) ** 2)

print "consistent mse left", mse_Left_cons
print "consistent mse rigt", mse_Right_cons  
            


cv2.namedWindow('disparity_L',cv2.WINDOW_NORMAL)
cv2.imshow('disparity_L',disparity_L)
cv2.namedWindow('disparity_R',cv2.WINDOW_NORMAL)
cv2.imshow('disparity_R',disparity_R)
cv2.namedWindow('disparity_L_cons',cv2.WINDOW_NORMAL)
cv2.imshow('disparity_L_cons',disparity_L_cons)
cv2.namedWindow('disparity_R_cons',cv2.WINDOW_NORMAL)
cv2.imshow('disparity_R_cons',disparity_R_cons)
cv2.waitKey(0)
cv2.destroyAllWindows()
                  

