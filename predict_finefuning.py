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
    x, y, n_class = iproc.load_labeled_imgs('./test/')

    '''
    推論
    '''
    predict = model.predict(x)

    '''
    結果の確認
    '''
    count = 0
    for idx in range(len(predict)):
        result = (idx, predict[idx].argmax(), y[idx])
        print('%d: estimate: %d, correct: %d'%result)
        if result[1] == result[2]: count += 1

    print('分類精度：%f ％'%(100 * count / len(predict)))
