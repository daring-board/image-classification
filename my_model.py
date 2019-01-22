import numpy as np
import img_processing as iproc
from keras.models import load_model

if __name__=="__main__":

    '''
    モデルのロード
    '''
    model = load_model('./model/custum_mobilenet.h5')

    '''
    画像のロード
    '''
    imgs, paths = iproc.load_imgs('./tmp/')
    labels = iproc.load_label()

    '''
    モデルによる推論
    '''
    predict = model.predict(imgs)

    '''
    結果の表示
    '''
    for idx in range(len(predict)):
        item, name = predict[idx], paths[idx]
        cls = np.argmax(item)
        print('Label: %s\t Name: %s'%(labels[cls], name))
