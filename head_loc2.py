# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.normalization import BatchNormalization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
input_dir = "/root/Documents/Kaggle/Fish_classification/train/original"
print(check_output(["ls", input_dir]).decode("utf8"))
img_dim_w = 48
img_dim_h = 48
# Any results you write to the current directory are saved as output.

import numpy as np
np.random.seed(1984)

import os
import glob
import tensorflow as tf
import cv2
import datetime
import pandas as pd
import time
import json
import warnings
warnings.filterwarnings("ignore")

from random import randint
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.metrics import mean_absolute_percentage_error, mean_absolute_error
#from sklearn.metrics import mean_absolute_error
from keras import __version__ as keras_version


def get_rect_coord():
    dir_ = '/root/Documents/Kaggle/Fish_classification/datasets/'
    #os.chdir(dir_)
    data = {}

    for file in glob.glob(dir_+"*.json"):
        with open(file) as data_file:
            data[file] = json.load(data_file)
    rects = {}
    for d in data:
        #print(data[d][0]['filename'])
        #print(data[d][0]['annotations'][0]['class'])
        img_ob = data[d]
        #print(len(img_ob))
        for i in range(0,len(img_ob)):
            #img_loc = imgs_dir + img_ob[i]['filename']
            f = 1 #1280
            base_name = os.path.basename(img_ob[i]['filename'])
            if len(img_ob[i]['annotations']) == 0:
                x0 = float(randint(0,1280))
                y0 = float(randint(0,760))
                xf = float(randint(0,1280))
                yf = float(randint(0,760))
                if x0 > xf:
                    a = x0
                    x0 = xf
                    xf = a
                if y0 > yf:
                    a = y0
                    y0 = yf
                    yf = a
                rects[base_name] = [x0,y0,xf,yf]
                #print(rects[base_name])
                continue
            annotation = img_ob[i]['annotations'][0]
            x0 = float(annotation['x'])
            y0 = float(annotation['y'])
            xf = x0 + float(annotation['width'])
            yf = y0 + float(annotation['height'])
            #print(x0,y0,xf,yf)
            #base_name = os.path.basename(img_ob[i]['filename'])
            rects[base_name] = [x0,y0,xf,yf]
    #print(rects)
    return rects

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (img_dim_w, img_dim_h), cv2.INTER_LINEAR) # 64, 64
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT'] # 'NoF'
    coords = get_rect_coord()
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'Fish_classification', 'train/original', fld, '*.jpg')
        #path = '/root/Documents/Kaggle/Fish_classification/train/train'
        #print(path)
        files = glob.glob(path)
        for fl in files:
            #print(fl)
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            if flbase in coords:
                #print(coords[flbase])
                y_train.append(coords[flbase])
            else:
                #print('not in coords')
                x0 = float(randint(0,1280))
                y0 = float(randint(0,760))
                xf = float(randint(0,1280))
                yf = float(randint(0,760))
                if x0 > xf:
                    a = x0
                    x0 = xf
                    xf = a
                if y0 > yf:
                    a = y0
                    y0 = yf
                    yf = a
                pos = [x0,y0,xf,yf]
                #print(pos)
                y_train.append(pos) #1280
            #y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('..', 'Fish_classification', 'test_stg1/test_stg1', '*.jpg')
    #path = '/root/Documents/Kaggle/Fish_classification/test_stg1/test_stg1/'
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['x0','y0','xf','yf'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    #train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255
    #print(test_id)

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def create_model():
    model = Sequential()
    #
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_dim_h, img_dim_w), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', $
    #
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    #model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', $
    #

    # Added layers
    ##model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    #model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', $
    #

    model.add(Flatten())
    model.add(Dense(96,init='he_uniform')) # 96,,
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.4))
    model.add(Dense(24, init='he_uniform')) # 24,,
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.2))
    model.add(Dense(4)) #8
    #model.add(BatchNormalization())
    #model.add(Activation('softmax'))

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
    model.compile(optimizer=sgd, loss='mae') #'mse'
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 48 # 24
    nb_epoch = 8 # 8
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    #print('kf\n',kf)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        #print(train_index, test_index)
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, verbose=0),
        ] #patience=3
        #print(X_train)
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
              callbacks=callbacks)
        #print(X_valid)
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        for i in range(len(predictions_valid)):
            print(predictions_valid[i],Y_valid[i])
            a = raw_input()
        score = mean_absolute_error(Y_valid, predictions_valid)
        #score = mean_absolute_error(Y_valid, predictions_valid)
        print('Score mae error: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("mae train independent avg: ", score)

    info_string = '_' + str(np.round(score,3)) + '_flds_' + str(nfolds) + '_eps_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 24 #24
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data() # <<
        #print(test_data)
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)
    #print(yfull_test) # cada yfull_test esta vindo repetido
    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 2
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
