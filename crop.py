import json
import glob, os
import cv2
from scipy import ndimage

dir_ = 'datasets/'
imgs_dir = '/root/Documents/Kaggle/Fish_classification/train/original/'
output_dir = '/root/Documents/Kaggle/Fish_classification/train/'
os.chdir(dir_)
data = {}
for file in glob.glob("*.json"):
    with open(file) as data_file:
        data[file] = json.load(data_file)

for d in data:
    #print(data[d][0]['filename'])
    #print(data[d][0]['annotations'][0]['class'])
    img_ob = data[d]
    for i in range(0,len(img_ob)):
        img_loc = imgs_dir + img_ob[i]['filename']
        image = cv2.imread(img_loc)
        for j in range(0,len(img_ob[i]['annotations'])):
            annotation = img_ob[i]['annotations'][j]
            x0 = int(annotation['x'])
            y0 = int(annotation['y'])
            xf = x0 + int(annotation['width'])
            yf = y0 + int(annotation['height'])
            #cropped = image[y0:yf, x0:xf]
            #if yf-y0 > xf-x0:
            #    cropped = ndimage.rotate(cropped, 90)
            cv2.rectangle(image,(x0,y0),(xf,yf),(0,255,0),2)
        out_loc = output_dir + img_ob[i]['class'] + '/' + img_ob[i]['filename']
        #print(out_loc)
        cv2.imwrite(out_loc + '_rect.jpg', image)
