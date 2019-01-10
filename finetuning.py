import img_processing as iproc
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout
from keras.applications.mobilenet import MobileNet
# from keras.applications.vgg16 import VGG16

if __name__=="__main__":

    '''
    学習済みモデルのロード(base_model)
    '''
    base_model = MobileNet(weights='imagenet', include_top=False)
    # base_model = VGG16(weights='imagenet', include_top=False)

    '''
    学習用画像のロード
    '''
    x, y = iproc.load_labeled_imgs('./train/')
    y = np_utils.to_categorical(y, 2)

    '''
    転移学習用のレイヤーを追加
    '''
    added_layer = GlobalAveragePooling2D()(base_model.output)
    added_layer = Dense(512, activation='relu')(added_layer)
    added_layer = Dropout(0.25)(added_layer)
    added_layer = Dense(2, activation='softmax', name='classification')(added_layer)

    '''
    base_modelと転移学習用レイヤーを結合
    '''
    model = Model(inputs=base_model.input, outputs=added_layer)
    model.summary()

    '''
    base_modelのモデルパラメタは学習させない。
    (added_layerのモデルパラメタだけを学習させる)
    '''
    for layer in base_model.layers:
         layer.trainable = False

    '''
    全体のモデルをコンパイル
    '''
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    '''
    モデルの学習
    '''
    model.fit(x, y, epochs=10)

    '''
    モデルパラメタの保存
    '''
    model.save('./model/custum_mobilenet.h5')
