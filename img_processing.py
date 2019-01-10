import os, sys, cv2
import numpy as np

def load_imgs(d_path):
    datas = process(d_path)
    return np.asarray(datas)

def load_labeled_imgs(d_path):
    data_dict = {}
    for d_name in os.listdir(d_path):
        path = d_path + d_name + '/'
        data = process(path)
        data_dict[d_name] = data
    x, y = [], []
    count = 0
    for key in data_dict.keys():
        for item in data_dict[key]:
            x.append(item)
            y.append(count)
        count += 1
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y

def process(d_path):
    datas = []
    for f_path in os.listdir(d_path):
        f = d_path + f_path
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
    return datas
