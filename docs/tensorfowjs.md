# TensorFlow.jsで使う
転移学習させた学習モデルをTensorFlow.jsを用いてjavascriptで使う。  
参照：https://js.tensorflow.org/tutorials/import-keras.html

## 準備
Python(Keras)で学習した学習モデルをそのまま使用できないので、  
下記のコマンドを実行して、学習モデルのパラメタファイルを変換する準備を行う。
```
pip install tensorflowjs
```

## 学習モデルを変換する
このプロジェクトでは、学習モデルのパラメタファイルは`./model/`ディレクトリに出力される。  
`./model/custum_mobilenet.h5`ファイルがあることを確認して以下を実行する。  
```
tensorflowjs_converter --input_format keras model\custum_mobilenet.h5 model
```  
実行後、`./model/`ディレクトリ内に`model.json`と`shard1of`というファイルが作成されていること
が確認出来たら成功である。

### 補足
MobileNetV2はTensorFlow.jsでは読み込めないモデル形式になっている。

## Node.jsなどのサーバ上で実行する
例えば、Node.jsでサーバを起動して、  
下記のようなHTMLファイルをエントリポイントとしてサイトを開いてみる。
```
<!-- Load TensorFlow.js. This is required to use MobileNet. -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/tensorflow/0.14.2/tf.js"> </script>
<!-- Load the MobileNet model. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@0.1.1"> </script>

<!-- Replace this with your image. Make sure CORS settings allow reading the image! -->
<img id="img" src="silvery_marmoset/n602.jpg" crossorigin="anonymous">
<div id="result"></div>

<!-- Place your code in the script tag below. You can also use an external .js file -->
<script>
  // Notice there is no 'import' statement. 'mobilenet' and 'tf' is
  // available on the index-page because of the script tag above.

  let img = tf.fromPixels(document.getElementById('img')).resizeNearestNeighbor([224,224]).toFloat();
  const offset = tf.scalar(255)
  img = img.div(offset).expandDims();
  const result = document.getElementById("result");

  // Load the model.
  async function loadModel(){
    model = await tf.loadModel('http://localhost:8001/model/model.json');
    console.log('loaded');
    return model
  }

  loadModel().then((model) => predict(model));

  async function predict(model){
    // Classify the image.
    let predictions = await model.predict(img).data();
    console.log('Predictions: ');
    console.log(predictions);
    let count = 0;
    predictions.forEach(x => {
       result.innerHTML += "Label: " + count + " Probablity: " + x + "<br/>";
       count += 1;
    });
  }
</script>

```
