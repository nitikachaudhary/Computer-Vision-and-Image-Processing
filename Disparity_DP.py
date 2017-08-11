import cv2
import numpy as np

left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)

height = left_img.shape[0]
width = left_img.shape[1]

#Disparity Computation for Left Image

#OcclusionCost = 20 (You can adjust this, depending on how much threshold you want to give for noise)
occlusion = 20

#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols

disparityMatrix = np.zeros((height,width),dtype=np.uint8)


# Now, its time to populate the whole Cost Matrix and DirectionMatrix and construct disparity Matrix

for k in range (0,height):
    costMatrix = np.zeros((width,width))
    directionMatrix = np.zeros((width,width))
    
    for i in range(0,width):
        costMatrix[i,0]=i*occlusion
        costMatrix[0,i]=i*occlusion
    for i in range(1,width):
        for j in range(1,width):
            min1=costMatrix[i-1,j-1]+abs(int(left_img[k,i])-int(right_img[k,j]))
            min2=costMatrix[i-1,j] + occlusion
            min3=costMatrix[i,j-1]+occlusion
            cmin=min(min1,min2,min3)
            costMatrix[i,j]=cmin
            if min1==cmin:
                directionMatrix[i,j]=1
            elif min2==cmin:
                directionMatrix[i,j]=2
            else:
                directionMatrix[i,j]=3
    
    p=width-1
    q=width-1
    
    while p!=0 and q!=0:
        val=directionMatrix[p,q]
        if val==1:
            disparityMatrix[k,p]=p-q
            p=p-1
            q=q-1
        elif val==2:
            p=p-1
        else:
            q=q-1


cv2.namedWindow('dis',cv2.WINDOW_NORMAL)
cv2.imshow('dis',disparityMatrix)
cv2.waitKey(0)
cv2.destroyAllWindows()
  
        
# Use the pseudocode from "A Maximum likelihood Stereo Algorithm" paper given as reference

