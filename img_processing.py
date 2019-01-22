import os, sys, cv2
import numpy as np

def load_imgs(d_path):
    datas, paths = process(d_path)
    return np.asarray(datas), paths

def load_labeled_imgs(d_path):
    data_dict, labels = {}, []
    for d_name in os.listdir(d_path):
        if d_name == 'empty': continue
        if d_name == '.DS_Store': continue
        path = d_path + d_name + '/'
        data, _ = process(path)
        data_dict[d_name] = data
        labels.append(d_name)
    x, y = [], []
    count = 0
    for key in data_dict.keys():
        for item in data_dict[key]:
            x.append(item)
            y.append(count)
        count += 1
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y, labels

def process(d_path):
    datas, paths = [], []
    for f_path in os.listdir(d_path):
        if f_path == 'empty': continue
        if f_path == '.DS_Store': continue
        if f_path == 'Thumbs.db': continue
        f = d_path + f_path
        paths.append(f)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        datas.append(img)
    return datas, paths
