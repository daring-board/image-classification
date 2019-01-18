# ImageClassification
image-classification repogitry for practice

## 概要
 1. とりあえず画像を分類してみる
 2. 分類器を自分好みにする
 3. 自分好みの分類器を検証する
 4. 補足(データについて)

## ソースコード
https://github.com/daring-board/image-classification

## 実行環境構築
実行環境にはAnacondaPythonを使用する。  
インストーラは下記に配置してある。ホームページよりダウンロードして実行してもよい。
```
\\micsvr03\Order\Develop\micstarter\PROJ\05_企画道場\02.活動\勉強会\20190111_静止画機械学習
Anaconda3-2018.12-Windows-x86_64.exe
```
インストール後の手順は以下。  
1. Anaconda Navigatorを起動して、左サイドバーメニューからEnviromentsを選択。  
![図1](./docs/0_1.png)
2. 展開された画面の環境選択セクションで、Createボタンをクリック。  
![図2](./docs/0_2.png)
3. 環境名をimage-classificationなど好きな名前を付与して、PackageをPython 3.6にして、Create。  
![図3](./docs/0_3.png)
4. 作成した環境名の[▼]ボタンを押下して、『Open Terminal』を選択。  
![図4](./docs/0_4.png)
5. 下記のコマンドを実行する。    
```
	pip install --ignore-installed --upgrade opencv-python  
	pip install --ignore-installed --upgrade tensorflow  
	pip install --ignore-installed --upgrade keras  
```

## プログラム実行方法  
0. ソースコードを取得する  
  下記のどちらかの方法により、ソースコードを取得して好きな場所に配置する。  
  + Gitコマンドが使える場合  
    Anacondaのターミナルでソースコードを取得したい場所をカレントディレクトリに変更して、以下を実行する。  
    ```
    git clone https://github.com/daring-board/image-classification.git  
    ```
  + Gitコマンドが使えない場合   
    `https://github.com/daring-board/image-classification よりZIPファイル` をダウンロードして展開   
1. とりあえず画像を分類してみる  
  1. 取得したプロジェクトフォルダ内のtmpフォルダに分類してみたい画像を配置する。  
  2. 下記のコマンドを実行する。    
  ```
  cd image-classification  
  python pretrained.py    
  ```
  補足：分類できる画像の種類は下記を参照。  
  `http://image-net.org/challenges/LSVRC/2012/browse-synsets`  
2. 分類器を自分好みにする  
  1. 学習データを下記からダウンロードする。  
  ```
  \\micsvr03\Order\Develop\micstarter\PROJ\05_企画道場\02.活動\勉強会\20190111_静止画機械学習\dataset\monkeys\8kind\full
  ```  
  2. ダウンロードデータを取得したプロジェクトフォルダに配置。  
  3. 下記コマンドでモデルを再学習する。  
  ```
  python finetuning.py
  ```  
3. 自分好みの分類器を検証する  
  1. 下記コマンドで再学習したモデルの精度を確認する。  
  ```
  python predict_finefuning.py
  ```  
4. 補足(データについて)  
  1. 少なすぎるデータで学習した場合のモデル精度を確認する。  
   1. 学習データを下記からダウンロードする。  
   ```
   \\micsvr03\Order\Develop\micstarter\PROJ\05_企画道場\02.活動\勉強会\20190111_静止画機械学習\dataset\monkeys\8kind\small
   ```  
   2. ダウンロードデータを取得したプロジェクトフォルダに配置。  
   3. 下記コマンドでモデルを再学習する。  
   ```
   python finetuning.py
   ```  
   4. 下記コマンドで再学習したモデルの精度を確認する。  
   ```
   python predict_finefuning.py
   ```  
  2. 不均一なデータで学習した場合のモデル精度を確認する。  
   1. 学習データを下記からダウンロードする。  
   ```
   \\micsvr03\Order\Develop\micstarter\PROJ\05_企画道場\02.活動\勉強会\20190111_静止画機械学習\dataset\monkeys\8kind\unbaranced
   ```   
   2. ダウンロードデータを取得したプロジェクトフォルダに配置。    
   3. 下記コマンドでモデルを再学習する。   
   ```
   python finetuning.py
   ```  
   4. 下記コマンドで再学習したモデルの精度を確認する。  
   ```
   python predict_finefuning.py
   ```  
