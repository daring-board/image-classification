import json, os
import random
import cv2
import numpy as np
import pandas as pd

from keras.utils import np_utils, Sequence
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

class DataSequence(Sequence):
    def __init__(self, data_path, label):
        self.batch = 1
        self.data_file_path = data_path
        self.datagen = ImageDataGenerator(
                            rotation_range=30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.5
                        )
        d_list = os.listdir(self.data_file_path)
        self.f_list = []
        for dir in d_list:
            if dir == 'empty': continue
            for f in os.listdir(self.data_file_path+'/'+dir):
                self.f_list.append(self.data_file_path+'/'+dir+'/'+f)
        self.label = label
        self.length = len(self.f_list)

    def __getitem__(self, idx):
        warp = self.batch
        aug_time = 3
        datas, labels = [], []
        label_dict = self.label

        for f in random.sample(self.f_list, warp):
            img = cv2.imread(f)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            datas.append(img)
            label = f.split('/')[2].split('_')[-1]
            labels.append(label_dict[label])
            # Augmentation image
            for num in range(aug_time):
                tmp = self.datagen.random_transform(img)
                datas.append(tmp)
                labels.append(label_dict[label])

        datas = np.asarray(datas)
        labels = pd.DataFrame(labels)
        labels = np_utils.to_categorical(labels, len(label_dict))
        return datas, labels

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        ''' 何もしない'''
        pass

if __name__=="__main__":
    shape = (224, 224, 3)
    input_tensor = Input(shape=shape)

    '''
    学習済みモデルのロード(base_model)
    '''
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    '''
    学習用画像のロード
    '''
    label_dict = {}
    count = 0
    for d_name in os.listdir('./train'):
        if d_name == 'empty': continue
        if d_name == '.DS_Store': continue
        d_name = d_name.split('_')[-1]
        label_dict[d_name] = count
        count += 1
    train_gen = DataSequence('./train', label_dict)

    '''
    転移学習用のレイヤーを追加
    '''
    added_layer = GlobalAveragePooling2D()(base_model.output)
    added_layer = Dense(1024)(added_layer)
    added_layer = BatchNormalization()(added_layer)
    added_layer = Activation('relu')(added_layer)
    added_layer = Dense(len(label_dict), activation='softmax', name='classification')(added_layer)

    '''
    base_modelと転移学習用レイヤーを結合
    '''
    model = Model(inputs=base_model.input, outputs=added_layer)

    '''
    base_modelのモデルパラメタは学習させない。
    (added_layerのモデルパラメタだけを学習させる)
    '''
    for layer in base_model.layers[:-3]:
         layer.trainable = False
    model.summary()

    '''
    全体のモデルをコンパイル
    '''
    opt = optimizers.Adam(lr=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    '''
    モデルの学習
    '''
    model.fit_generator(
         train_gen,
         epochs=25,
         steps_per_epoch=int(train_gen.length),
        #  validation_data=train_gen,
        #  validation_steps=4*int(train_gen.length / 10),
    )

    '''
    モデルパラメタの保存
    '''
    model.save('./model/custum_mobilenet.h5')

    '''
    ラベル情報を保存
    '''
    l_dict = {label_dict[name]: name for name in label_dict.keys()}
    json.dump(l_dict, open('./model/labels.json', 'w'), indent=4)
