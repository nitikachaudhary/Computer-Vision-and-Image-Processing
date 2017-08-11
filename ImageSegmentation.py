import cv2
import numpy as np
import random
from copy import deepcopy

#reading image
img = cv2.imread('Butterfly.jpg')


#reading shape of image
height = img.shape[0]
width = img.shape[1]

#Initializing segmented image and feature matrix
final_img = np.zeros((height,width,3),dtype=np.uint8)
feature_mat = np.zeros((height*width,5))

#Constructing the feature matrix
k=0
for i in range (0,height):
    for j in range(0,width):
        intensity=img[i,j]
        feature_mat[k]=[intensity[0],intensity[1],intensity[2],i,j]
        k=k+1

#cluster1 will contain the points within the cluster of initialized mean
#cluster2 will contain the ones lying outside
cluster1=[]
cluster2=[]

#threshold values for euclidean distance
h=90
iter_val=10

#mean shift segmentation algorithm
seed_point=feature_mat[0]
while (len(feature_mat)>0):
    for i in range (0,len(feature_mat)):
        euc_dis=np.sqrt(np.sum((seed_point-feature_mat[i])**2))
        if euc_dis<h:
            cluster1.append(feature_mat[i])
        else:
            cluster2.append(feature_mat[i])
            
    mean = np.mean(cluster1, axis=0)
    distance = np.sqrt(np.sum((mean-seed_point)**2))
    
    if distance < iter_val:
        for i in range (0,len(cluster1)):
            final_img[int(cluster1[i][3]),int(cluster1[i][4])] =mean[0:3]
        feature_mat=deepcopy(cluster2)
        if(len(feature_mat)==0):
            break
        seed_point=random.choice(feature_mat)
    else:
        seed_point=mean
    cluster1[:] = []
    cluster2[:] = []
        


cv2.namedWindow('Butterfly',cv2.WINDOW_NORMAL)
cv2.imshow('Butterfly',img)
cv2.namedWindow('Butterfly_Segments',cv2.WINDOW_NORMAL)
cv2.imshow('Butterfly_Segments',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()