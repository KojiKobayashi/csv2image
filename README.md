# これは何？
編み物とかで使うことがあるやつ

# 使い方
* 最新版をダウンロード（READMEの更新不要な固定リンク）
  * Release 一覧: https://github.com/KojiKobayashi/csv2image/releases
  * 最新 Release: https://github.com/KojiKobayashi/csv2image/releases/latest
  * Windows版zip: https://github.com/KojiKobayashi/csv2image/releases/latest/download/Image2nitting-windows.zip
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
* いらない色を選択して再実行
* セルの線を可変に