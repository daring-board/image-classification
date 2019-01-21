import json
import img_processing as iproc
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2

if __name__=="__main__":

    '''
    学習済みモデルのロード(base_model)
    '''
    # base_model = MobileNetV2(weights='imagenet', include_top=False)
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    # base_model = Xception(weights='imagenet', include_top=False)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    '''
    学習用画像のロード
    '''
    x, y, labels = iproc.load_labeled_imgs('./train/')
    y = np_utils.to_categorical(y, len(labels))

    '''
    転移学習用のレイヤーを追加
    '''
    added_layer = GlobalAveragePooling2D()(base_model.output)
    added_layer = Dense(1024, activation='relu')(added_layer)
    added_layer = Dropout(0.5)(added_layer)
    added_layer = Dense(len(labels), activation='softmax', name='classification')(added_layer)

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
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    '''
    モデルの学習
    '''
    model.fit(x, y, epochs=50)

    '''
    モデルパラメタの保存
    '''
    model.save('./model/custum_mobilenet.h5')

    '''
    ラベル情報を保存
    '''
    l_dict = {idx: labels[idx] for idx in range(len(labels))}
    json.dump(l_dict, open('./model/labels.json', 'w'), indent=4)
