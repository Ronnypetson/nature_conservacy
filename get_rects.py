import json
import glob, os

dir_ = 'datasets/'
os.chdir(dir_)
data = {}
def get_rect_coord():
    for file in glob.glob("*.json"):
        with open(file) as data_file:
            data[file] = json.load(data_file)
    rects = {}
    for d in data:
        #print(data[d][0]['filename'])
        #print(data[d][0]['annotations'][0]['class'])
        img_ob = data[d]
        for i in range(0,len(img_ob)):
            #img_loc = imgs_dir + img_ob[i]['filename']
            if len(img_ob[i]['annotations']) == 0:
                continue
            annotation = img_ob[i]['annotations'][0]
            x0 = int(annotation['x'])
            y0 = int(annotation['y'])
            xf = x0 + int(annotation['width']) + 26
            yf = y0 + int(annotation['height']) + 26
            base_name = os.path.basename(img_ob[i]['filename'])
            rects[base_name] = [x0,y0,xf,yf]
    return rects
