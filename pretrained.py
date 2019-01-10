import img_processing as iproc
from keras.applications.mobilenet import MobileNet, decode_predictions

if __name__=="__main__":

    '''
    学習済みモデルのロード
    '''
    model = MobileNet(weights='imagenet')
    # model.summary()

    '''
    学習用画像のロード
    '''
    imgs = iproc.load_imgs('./tmp/')

    '''
    学習済みモデルによる推論
    '''
    predict = model.predict(imgs)
    predict = decode_predictions(predict, top=1)

    '''
    分類ラベルリスト
    http://image-net.org/challenges/LSVRC/2012/browse-synsets
    '''
    for item in predict:
        print('Predicted:', item)
