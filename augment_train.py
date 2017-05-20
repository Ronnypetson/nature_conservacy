import imgaug as ia
from imgaug import augmenters as iaa
import glob, os, cv2
st = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    st(iaa.Affine(
            scale={"x": (0.2, 0.7), "y": (0.2, 0.7)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            #order=ia.ALL, # use any of scikit-image's interpolation methods
            #cval=(0, 1.0), # if mode is constant, use a cval between 0 and 1.0
            #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    #iaa.Crop(px=(0, 40)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
dir = '/root/Documents/Kaggle/Fish_classification/train/cropped/'
os.chdir(dir)
folders = ['OTHER', 'SHARK', 'YFT'] #'ALB','ALB', 'BET', 'DOL', 'LAG', 'NoF'
for f in folders:
    os.chdir(f)
    for file in glob.glob('*.jpg'):
        img = cv2.imread(file)
        for i in range(5):
            img_aug = seq.augment_images([img])
            cv2.imwrite(file + 'aug' + str(i) + '.jpg', img_aug[0])
    os.chdir('..')
