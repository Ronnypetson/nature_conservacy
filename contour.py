import numpy as np
import glob, os
import cv2

dir_ = 'train/train'
os.chdir(dir_)
folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
for f in folders:
    os.chdir(f)
    for file in glob.glob('*.jpg'):
        #print(file)
        img = cv2.imread(file)
        #rows,cols = img.shape
        #M = np.float32([[1,0,100],[0,1,50]])
        #img = cv2.medianBlur(img,5)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV) #0
        img2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #img2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img2, contours, -1, (0,255,0), 3)
        cv2.imwrite('../../contour/'+f+'/'+file+'_contour.jpg', img2) #img2
    os.chdir('..')
