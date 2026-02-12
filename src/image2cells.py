import cv2
import numpy as np
import sys

import settings as cfg

cell_height, cell_width = cfg.cell_height, cfg.cell_width
line_thickness = cfg.line_thickness
thick_line_thickness = cfg.thick_line_thickness
thick_line_interval = cfg.thick_line_interval
background_color = cfg.cell_line_color
colors_number = cfg.number_of_colors
number_of_line_cells = cfg.number_of_line_cells
denoise = cfg.denoise

# TODO: idx 画像でのみ可能
# ノイズ除去　3×3のopening
def _remove_noise(image):
    # denoised_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # return denoised_image
    return image

# メディアンカット法で色削減
def _median_cut(image):
    z = image.reshape((-1, 3))
    z = np.float32(z)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = colors_number  # Number of colors
    
    _, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    if labels is None or centers is None:
        raise ValueError("K-means clustering failed to produce labels or centers.")

    # Convert back to uint8 and make original image
    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape((image.shape))
    return quantized_image, centers


# 縦横比を保って画像をリサイズ
def _resize_image(image, new_width, src_height, src_width):
    if new_width <= 0:
        raise ValueError("new_width and new_height must be positive integers.")
    
    resize_ratio = new_width / src_width
    new_height = int(src_height * resize_ratio)
    new_size = (new_width, new_height)
    print(f"Resizing image to: {new_size}, original size: ({src_width}, {src_height})")
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    return resized_image


# 画像を縦に引き伸ばす
def _resize_image_2slim(image):
    if cell_height <= 0 or cell_width <= 0:
        raise ValueError("cell_height and cell_width must be positive integers.")
    
    height, width = image.shape[:2]
    new_size = (int(width), int((height * cell_width) / cell_height))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return resized_image


# 引数で画像ファイルを受け取り、画像を表示する関数
def display_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 画像のピクセル間の線を引く
def imageToPixelize(image):
    # data1の各ピクセルに対応するRGB値を取得し、カラー画像を作成
    height, width = image.shape[:2]

    # 出力用画像に変換
    # 1ピクセルは33×27ピクセルに拡大。拡大したピクセルをセルと呼ぶ。
    # セルの間は1ピクセルの黒線で区切る。さらに黒線は5セルごとに3ピクセルの太さになる。
    if thick_line_interval <= 0:
        raise ValueError("thick_line_interval must be a positive integer.")

    out_height = height * cell_height + (height // thick_line_interval) * (thick_line_thickness - line_thickness) + (height - 1) * line_thickness
    out_width = width * cell_width + (width // thick_line_interval) * (thick_line_thickness - line_thickness) + (width - 1) * line_thickness
    output_image = np.full((out_height, out_width, 3), background_color, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            start_y = i * (cell_height + line_thickness) + (i // thick_line_interval) * (thick_line_thickness - line_thickness)
            start_x = j * (cell_width + line_thickness) + (j // thick_line_interval) * (thick_line_thickness - line_thickness)
            output_image[start_y:start_y + cell_height, start_x:start_x + cell_width] = image[i, j]

    return output_image

# csv読み込み
# 系統,色番,R,G,B,コメント
# 一行目はヘッダー
def load_color_csv(file_path):
    colors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            system, color_number, r, g, b, _ = parts
            rgb = [int(r), int(g), int(b)]
            colors.append(({"type": system, "color_number": color_number, "RGB": rgb}))
    return colors

def nearest_color(target_rgb, color_list):
    min_distance = float('inf')
    nearest = None
    for color in color_list:
        r, g, b = color["RGB"]
        distance = (float(target_rgb[0]) - float(r)) ** 2
        distance += (float(target_rgb[1]) - float(g)) ** 2
        distance += (float(target_rgb[2]) - float(b)) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest = color
    return nearest

def map_colors_to_palette(centers, palette):
    mapped_colors = []
    for center in centers:
        nearest = nearest_color(list(reversed(center)), palette)
        mapped_colors.append(nearest)
    return mapped_colors

def count_color_pixels(grey):
    color_counts = {}
    height, width = grey.shape[:2]
    for i in range(height):
        for j in range(width):
            idx = grey[i, j]
            if idx in color_counts:
                color_counts[idx] += 1
            else:
                color_counts[idx] = 1

    ret = [0] * len(color_counts)
    for idx, count in color_counts.items():
        ret[idx] = count
    return ret

def map_image_colors_to_palette(image, centers, palette):
    # 画像をcentersのインデックスの1channel画像に変換
    grey = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            target_rgb = image[i, j]
            for idx, center in enumerate(centers):
                if np.array_equal(target_rgb, center):
                    grey[i, j] = idx
                    break
                grey[i, j] = 255  # centersにない色は255にする
    
    # 色ごとの数
    color_counts = count_color_pixels(grey)

    mapped_colors = map_colors_to_palette(centers, palette)
    # for idx, color in enumerate(mapped_colors):
    #     print(f"Center {idx}:  RGB {color['RGB']}, center RGB {list(reversed(centers[idx]))}")

    height, width = grey.shape[:2]
    mapped_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            idx = grey[i, j]
            if idx < len(mapped_colors):
                mapped_image[i, j] = list(reversed(mapped_colors[idx]["RGB"]))
            else:
                mapped_image[i, j] = [255, 255, 255]  # centersにない色は白にする
    return mapped_image, mapped_colors, color_counts


def run_image(src):
    resize = _resize_image_2slim(src)
    median, centers = _median_cut(resize)
    noise = _remove_noise(median)

    # noise の色数を出力
    unique_colors = np.unique(median.reshape(-1, median.shape[2]), axis=0)
    print(f"Number of unique colors after quantization and denoising: {len(unique_colors)}")
    unique_colors = np.unique(noise.reshape(-1, noise.shape[2]), axis=0)
    print(f"Number of unique colors after quantization and denoising: {len(unique_colors)}")

    dst = _resize_image(noise, number_of_line_cells, src.shape[0], src.shape[1])

    palette = load_color_csv("data/merinorainbow.csv")
    mapped_image, mapped_colors, color_counts = map_image_colors_to_palette(dst, centers, palette)

    if denoise:
        mapped_image = _remove_noise(mapped_image)

    pixel = imageToPixelize(mapped_image)

    return pixel, centers,  mapped_colors, color_counts

def run(filen_name):
    src = cv2.imread(filen_name)
    if src is None:
        raise ValueError("Image not found or unable to load.")

    return run_image(src)


# 引数でファイル名を受け取る
if __name__ == "__main__":
    # 第一引数から画像パス取得
    if len(sys.argv) < 2:
        raise ValueError("Usage: python image2cells.py <image_path>")
    file_path = sys.argv[1]

    pixels, centers, mapped_colors, color_counts = run(file_path)
    cv2.imwrite("output_pixelized.png", pixels)
    print("Centers:", centers)
    print("mapped_colors:", mapped_colors)
    print("Color Counts:", color_counts)

    display_image(pixels)

    #create_processing_image()