# import cv2
# import numpy as np

# img0=cv2.imread("d:\\Users\\SaniFaust-LEGION\\Pictures\\image0.jpg")

# #中值濾波
# img_median=cv2.medianBlur(img0,5)
# font = cv2.FONT_HERSHEY_SIMPLEX

import numpy as np
import cv2
#椒盐噪声
def saltpepper(img,n): #n为噪声比例
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m): #随机取值赋值为255（白色）
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2: #单通道
            img[j,i]=255
        elif img.ndim==3: #三通道
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m): #随机取值赋值为0（黑色）
        i=int(np.random.random()*img.shape[1]) #image.shape[0]获取图像的高度
        j=int(np.random.random()*img.shape[0]) #image.shape[1]获取图像的宽度
        if img.ndim==2:                         #image.shape[2]获取图像的通道数
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img
img=cv2.imread('d:\\Users\\SaniFaust-LEGION\\Pictures\\image0.png')
cv2.imshow('origial',img)
saltImage=saltpepper(img,0.1)
cv2.imshow('saltImage',saltImage)
cv2.imwrite('saltImage.png',saltImage)

#使用自带中值滤波函数
#medianBlur = cv2.medianBlur(img, 3)
#cv2.imshow('img_median', medianBlur)

#中值滤波函数实现
img_copy = cv2.imread('saltImage.png')
for i in range(0,saltImage.shape[0]):
   for j in range(0,saltImage.shape[1]):#取维度
       img_copy[i][j] = saltImage[i][j]
#用3*3的中值滤波器
step=3
def median_filter(x,y):
    sum_s=[]
    for k in range(-int(step/2),int(step/2)+1):
        for m in  range(-int(step/2),int(step/2)+1):
            sum_s.append(saltImage[x+k,y+m,1])
    sum_s.sort()
    return sum_s[(int(step*step/2)+1)] #取中值
for i in range(int(step/2),saltImage.shape[0]-int(step/2)):
    for j in range(int(step/2),img.shape[1]-int(step/2)):
        img_copy[i][j] = median_filter(i,j)

cv2.imshow('img_median', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()