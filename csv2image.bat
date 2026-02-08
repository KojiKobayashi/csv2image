@REM 第一引数 画像のcsvファイルパス
@REM 第二引数 インデックスとRGB値の対応表csvファイルパス
@REM 第三引数 出力画像ファイルパス(jpg/png/bmpはいけるはず)

@echo off
cd /d %~dp0

uv run src\csv2image.py %1 %2 %3