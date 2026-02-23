# これは何？
編み物とかで使うことがあるやつ

# 使い方

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
* 色を塗る箇所でundo
* 色を塗る箇所でスポイト機能