# これは何？
編み物とかで使うことがあるやつ

# 使い方

## 環境設定
* なんとかしてpythonとuvを入れる
* どっかにフォルダを作って内容物をコピー
* `uv venv` で仮想環境作成
* `uv pip install -r requirements.txt` で必要リブラり取得
* コマンドプロンプトから `.\csv2image.bat "src\in1.csv" "src\in2.csv" "test.jpg"` でtest.jpgが「ぺ」になったら成功

## 使用方法
` .\csv2image.bat csvファイルパス1 CSVファイルパス2 出力画像パス`

* 第一引数 画像のcsvファイルパス
* 第二引数 インデックスとRGB値の対応表csvファイルパス
* 第三引数 出力画像ファイルパス(jpg/png/bmpはいけるはず)

## 実行設定
src/settings.py をいじると何とかなる

## TODO
* 糸との紐づけ
* 色の出力
* 矩形を入力してもらう
* 色ごとの画素数数える