import cv2
import numpy as np

import settings as cfg


# 引数でcsvファイル二つを受け取り、画像として保存する関数
def csv_to_image(csv_file1, csv_file2, output_filename):
    # CSVファイルを読み込み、NumPy配列に変換
    # csv は8bitのグレースケール画像で、一要素一ピクセルを想定
    data1 = np.loadtxt(csv_file1, delimiter=',', dtype=np.uint8)
    if min(data1.shape) <= 1:
        raise ValueError("csv_file1 must represent a 2D image.")

    # csv_file2は、4カラム。一カラム目がピクセルのインデックス、二カラム目以降がそのピクセルのRGB値を示す。
    data2 = np.loadtxt(csv_file2, delimiter=',', dtype=np.uint8)
    
    # data2をインデックスとRGBのハッシュに変換
    index_to_rgb = {row[0]: row[1:4] for row in data2}

    # data1の各ピクセルに対応するRGB値を取得し、カラー画像を作成
    height, width = data1.shape
    default_color = cfg.no_index_color
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixel_index = data1[i, j]
            if pixel_index in index_to_rgb:
                # RBGをBGRにして代入
                color_image[i, j] = index_to_rgb[pixel_index][::-1]
            else:
                color_image[i, j] = default_color
    
    # 出力用画像に変換
    # 1ピクセルは33×27ピクセルに拡大。拡大したピクセルをセルと呼ぶ。
    # セルの間は1ピクセルの黒線で区切る。さらに黒線は5セルごとに3ピクセルの太さになる。
    cell_height, cell_width = cfg.cell_height, cfg.cell_width
    line_thickness = cfg.line_thickness
    thick_line_thickness = cfg.thick_line_thickness
    thick_line_interval = cfg.thick_line_interval

    out_height = height * cell_height + (height // thick_line_interval) * (thick_line_thickness - line_thickness) + (height - 1) * line_thickness
    out_width = width * cell_width + (width // thick_line_interval) * (thick_line_thickness - line_thickness) + (width - 1) * line_thickness
    output_image = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            start_y = i * (cell_height + line_thickness) + (i // thick_line_interval) * (thick_line_thickness - line_thickness)
            start_x = j * (cell_width + line_thickness) + (j // thick_line_interval) * (thick_line_thickness - line_thickness)
            output_image[start_y:start_y + cell_height, start_x:start_x + cell_width] = color_image[i, j]

    # 画像を保存
    cv2.imwrite(output_filename, output_image)

# 引数でファイル名を受け取る
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python csv2image.py <csv_file1> <csv_file2> <output_image>")
        sys.exit(1)
    
    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    output_filename = sys.argv[3]
    
    csv_to_image(csv_file1, csv_file2, output_filename)
