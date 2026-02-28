# merinorainbow 補助ツール

このフォルダは `merinorainbow` 商品向けのASIN生成ツールです。
アプリ本体とは独立して使います。

## 前提

- `uv pip install -r requirements-dev.txt`
- `python -m playwright install chromium`

## 1) 色番リストから ASIN CSV を再生成

入力: `merinorainbow_color_numbers.txt`（1行1色番）

実行:

```powershell
python .\scripts\build_merinorainbow_asin_csv.py --engine auto
```

主な出力:

- `scripts/merinorainbow_asin_generated.csv`（`色番,ASIN,Amazonリンク`）
- `scripts/merinorainbow_color_asin_map.json`（`色番 -> ASIN`）

確認のみ（書き込みなし）:

```powershell
python .\scripts\build_merinorainbow_asin_csv.py --dry-run
```

ブラウザ表示で確認:

```powershell
python .\scripts\build_merinorainbow_asin_csv.py --engine playwright --show-browser --dry-run
```

## 2) data 置換向け CSV を作る

`data/merinorainbow.csv` の形式（`系統,色番,R,G,B,ASIN`）に合わせた置換用CSVを作ります。

```powershell
python .\scripts\merge_merinorainbow_asin_to_data_csv.py
```

出力:

- `scripts/merinorainbow_for_data_replace.csv`

このファイルを手動で `data/merinorainbow.csv` に置き換えてください。
