import cv2
import glob, os
import numpy as np

target_dir = '/root/Documents/Kaggle/Fish_classification/test_stg1/test_stg1_/*.jpg'
output_dir = '/root/Documents/Kaggle/Fish_classification/test_stg1/marked'
coords_path = './submission_loss__41.546_flds_2_eps_40_fl_96_folds_2_2017-03-12-22-33.csv'

coord = {}
Y = np.loadtxt(open(coords_path, "rb"), delimiter=",", dtype=np.str, skiprows=1, usecols=[4])
X = np.loadtxt(open(coords_path, "rb"), delimiter=",", skiprows=1, usecols=range(4))
for i in range(len(X)):
    coord[Y[i]] = [X[i][0],X[i][1],X[i][2],X[i][3]]

for file in glob.glob(target_dir):
    img = cv2.imread(file)
    base_name = os.path.basename(file)
    c = coord[base_name]
    x0 = int(float(c[0]))
    y0 = int(float(c[1]))
    x1 = int(float(c[2]))
    y1 = int(float(c[3]))
    if x0 > x1:
        a = x0
        x0 = x1
        x1 = a
    if y0 > y1:
        a = y0
        y0 = y1
        y1 = a
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 3)
    cv2.imwrite(output_dir+'/'+base_name+'_rect.jpg',img)
