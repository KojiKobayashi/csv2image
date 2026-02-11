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
denoise = cfg.denoise


# ノイズ除去　3×3のメディアンフィルタ
def _remove_noise(image):
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image

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


def run_image(src):
    resize = _resize_image_2slim(src)
    median, centers = _median_cut(resize)
    noise = _remove_noise(median)
    dst = _resize_image(noise, 120, src.shape[0], src.shape[1])

    if denoise:
        dst = _remove_noise(dst)

    pixel = imageToPixelize(dst)

    return pixel, centers

def run(filen_name):
    src = cv2.imread(filen_name)
    if src is None:
        raise ValueError("Image not found or unable to load.")

    return run_image(src)

# def create_processing_image():
#     '''
#     1.UIからユーザーに画像を選択させる
#     2.選択された画像を表示
#     3.画像上でユーザーに矩形を一つ囲わせる
#     4.囲まれた矩形部分を切り出し、run関数に渡して処理を行う
#     '''
#     # 1. UIからユーザーに画像を選択させる
#     from tkinter import Tk
#     from tkinter import filedialog
#     root = Tk()
#     root.withdraw()  # メインウィンドウを表示しない
#     file_path = filedialog.askopenfilename(title="画像を選択してください", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
#     root.destroy()
#     if not file_path:
#         raise ValueError("No file selected.")
    
#     # 2. 選択された画像を表示
#     src = cv2.imread(file_path)
#     if src is None:
#         raise ValueError("Image not found or unable to load.")

#     # 3. rootのメインウィンドウに画像を表示して、画像上でユーザーに矩形を一つ囲ませる
#     r = cv2.selectROI("Select ROI", src, fromCenter=False, showCrosshair=True)
#     cv2.destroyWindow("Select ROI")
#     if r == (0,0,0,0):
#         r = (0, 0, src.shape[1], src.shape[0])  # 全体を選択したことにする
#     roi = src[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

#     # UI作成
#     from tkinter import simpledialog, Canvas, Scrollbar
#     root = Tk()
#     canvas = Canvas(
#         root,
#         width=400,
#         height=300,
#     )
#     canvas.pack()

#     display_image(roi)

#     # canvasにroiを表示
#     roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi_byte = cv2.imencode('.png', roi_rgb)[1].tobytes()

#     canvas.create_image(0, 0, anchor='nw', image=roi_byte)
#     canvas.config(scrollregion=canvas.bbox("all"))

#     scrollbar = Scrollbar(root, command=canvas.yview)
#     scrollbar.pack(side='right', fill='y')


#     display_image(roi)
    
#     root.destroy()


#     # for debug
#     cv2.rectangle(src, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), (0,255,0), 2)
#     display_image(src)

# 引数でファイル名を受け取る
if __name__ == "__main__":
    # 第一引数から画像パス取得
    if len(sys.argv) < 2:
        raise ValueError("Usage: python image2cells.py <image_path>")
    file_path = sys.argv[1]

    pixels, centers = run(file_path)
    cv2.imwrite("output_pixelized.png", pixels)
    print("Centers:", centers)

    display_image(pixels)

    #create_processing_image()