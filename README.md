# これは何？
編み物とかで使うことがあるやつ

# 使い方
* 最新版をダウンロード
Image2nitting-windows:
https://github.com/KojiKobayashi/csv2image/actions/runs/22517283179
* zip を適当なフォルダに展開
* Image2nitting.exe　をダブルクリック
  * windowsが警告を出すかもしれない
* ブラウザが開くので作業開始


# 以下開発版の実行方法

## 環境設定
* なんとかしてpythonとuvを入れる
* どっかにフォルダを作って内容物をコピー
* `uv venv` で仮想環境作成
* `uv pip install -r requirements.txt` で必要リブラり取得

## 使用方法
* コマンドプロンプトから

```.\.venv\Scripts\activate
uv run streamlit run .\app\app.py
```

# TODO
* 色数を減らして再実行
* アスペクト比を保って矩形選択
* 出力糸のリンクなどを追加
* いらない色を選択して再実行
* セルの線を可変に
* ノイズ除去を横方向の1pixel除去にしてはどうか