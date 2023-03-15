import numpy as np
import cv2 as cv
import random as rd
img = cv.imread('123.jpeg')
type(img)
img.shape
_blue = img[1332,1999,0]
_green = img[500,600,1]
_red = img[500,600,2]

def showbgr(A,y,x):
    print(f'blue {A[y,x,0]}')
    print(f'green {A[y,x,1]}')
    print(f'red {A[y,x,2]}')

blue = [[] for i in range(1332)]
for y in range(1332):
    for x in range(1999):
        blue[y].append(img[y,x,0])

type(img)














#for x in range(500,600):
    #for y in range(1500,2000):
        #img[x,y]=0
#cv.imshow('intermediate', img)

# 按下任意鍵則關閉所有視窗
#cv.waitKey(0)
#cv.destroyAllWindows()