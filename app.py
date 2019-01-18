import cv2, json, os
import numpy as np
import configparser
import tensorflow as tf
from flask import Flask, jsonify, request, render_template, redirect, url_for, send_from_directory
from threading import Thread
from keras.models import Sequential, load_model


app = Flask(__name__)

UPLOAD_FOLDER = './upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
graph = tf.get_default_graph()
with graph.as_default():
    model = load_model('./model/custum_mobilenet.h5')


@app.route('/', methods = ["GET", "POST"])
def root():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        f = request.files['FILE']
        f_path = save_img(f)
        predict = predict_core([f_path]).data.decode('utf-8')
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        return render_template(
                'index.html',
                filepath=path,
                predict=json.loads(predict)['data'][0]
            )


@app.route('/upload/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def save_img(f):
    stream = f.stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    f_path = UPLOAD_FOLDER+'/'+f.filename
    cv2.imwrite(f_path, img)
    return f_path

def predict_core(path_list):
    global model
    global graph
    data = preprocess(path_list)
    names = [item.split('/')[-1] for item in path_list]
    with graph.as_default():
        pred_class = model.predict(data)
    label_dict = json.load(open('./model/labels.json', 'r'))
    result = []
    for idx in range(len(data)):
        top3 = pred_class[idx].argsort()[::-1][:2]
        item = {
            'name': names[idx],
            'class1': (label_dict[str(top3[0])], str(pred_class[idx][top3[0]])),
            'class2': (label_dict[str(top3[1])], str(pred_class[idx][top3[1]])),
        }
        result.append(item)
    print(result)

    return jsonify({
            'status': 'OK',
            'data': result
        })

def preprocess(f_list):
    datas = []
    for f_path in f_list:
        print(f_path)
        img = cv2.imread(f_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
    datas = np.asarray(datas)
    return datas

def abortWithInvalidParams(reason, debug={}):
    abort(400, {
        'errorCode': 1,
        'description': 'invalid params',
        'reason': reason,
        'debug': debug,
    })


def abortWithNoItem(reason, debug={}):
    abort(404, {
        'errorCode': 2,
        'description': 'no item',
        'reason': reason,
        'debug': debug,
    })


def abortWithServerError(reason, debug={}):
    abort(500, {
        'errorCode': 3,
        'description': 'server error',
        'reason': reason,
        'debug': debug,
    })

Thread(target=load_model, daemon=True).start()
if __name__ == "__main__":
    print(" * Flask starting server...")
    app.run(host='0.0.0.0',port=5000)
