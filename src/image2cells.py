import cv2
import numpy as np
import sys
from typing import List

import settings as cfg


class Color:
    """色を表現するクラス"""
    def __init__(self, type: str, color_number: str, rgb: list, lab: list):
        self.type = type
        self.color_number = color_number
        self.rgb = rgb
        self.lab = lab

    def __repr__(self):
        return f"Color(type='{self.type}', number='{self.color_number}', rgb={self.rgb})"

class ColorCount(Color):
    """色とそのピクセル数を表現するクラス"""
    def __init__(self, type: str, color_number: str, rgb: list, lab: list, count: int):
        super().__init__(type, color_number, rgb, lab)
        self.count = count

    def __repr__(self):
        return f"ColorCount(type='{self.type}', number='{self.color_number}', rgb={self.rgb}, count={self.count})"

class ImageToPixels:
    """画像をドット絵（ピクセルアート）に変換するクラス"""
    
    def __init__(self):
        """cfg から設定値を初期化"""
        self._cell_height = cfg.cell_height
        self._cell_width = cfg.cell_width
        self._line_thickness = cfg.line_thickness
        self._thick_line_thickness = cfg.thick_line_thickness
        self._thick_line_interval = cfg.thick_line_interval
        self._background_color = cfg.cell_line_color
        self._colors_number = cfg.number_of_colors
        self._number_of_line_cells = cfg.number_of_line_cells
        self._denoise = cfg.denoise

    # ==================== Properties ====================
    @property
    def cell_height(self):
        return self._cell_height

    @cell_height.setter
    def cell_height(self, value):
        self._cell_height = value

    @property
    def cell_width(self):
        return self._cell_width

    @cell_width.setter
    def cell_width(self, value):
        self._cell_width = value

    @property
    def line_thickness(self):
        return self._line_thickness

    @line_thickness.setter
    def line_thickness(self, value):
        self._line_thickness = value

    @property
    def thick_line_thickness(self):
        return self._thick_line_thickness

    @thick_line_thickness.setter
    def thick_line_thickness(self, value):
        self._thick_line_thickness = value

    @property
    def thick_line_interval(self):
        return self._thick_line_interval

    @thick_line_interval.setter
    def thick_line_interval(self, value):
        self._thick_line_interval = value

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        self._background_color = value

    @property
    def colors_number(self):
        return self._colors_number

    @colors_number.setter
    def colors_number(self, value):
        self._colors_number = value

    @property
    def number_of_line_cells(self):
        return self._number_of_line_cells

    @number_of_line_cells.setter
    def number_of_line_cells(self, value):
        self._number_of_line_cells = value

    @property
    def denoise(self):
        return self._denoise

    @denoise.setter
    def denoise(self, value):
        self._denoise = value


    # ==================== Private Methods ====================
    def _remove_noise_ori(self, image:np.ndarray) -> np.ndarray:
        '''
        _remove_noise_ori：今のとこ未実装
        
        :param self: 説明
        :param image: 説明
        '''
        return image


    def _remove_noise(self, image:np.ndarray) -> np.ndarray:
        """TODO: 強すぎるのであまりやらないほうがいい、ノイズ除去 3×3のopening"""
        denoised_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return denoised_image


    def _median_cut(self, image:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """メディアンカット法で色削減"""
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        z = np.float32(lab_image)
        z = z.reshape(-1, 3)
        z[:, 0] = z[:, 0] * (100.0 / 255.0)  # L を0~100に正規化
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = self._colors_number
        
        bestLabels = np.empty((z.shape[0], 1), dtype=np.int32)
        _, labels, centers = cv2.kmeans(z, k, bestLabels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if labels is None or centers is None:
            raise ValueError("K-means clustering failed to produce labels or centers.")
        
        centers = centers.reshape(-1, 3)
        centers[:, 0] = centers[:, 0] * (255.0 / 100.0)  # L を0~255に戻す
        centers = np.uint8(centers)
        centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_LAB2BGR).reshape(-1, 3)
        
        quantized_image = centers[labels.flatten()]
        quantized_image = quantized_image.reshape((image.shape))
        return quantized_image, labels.reshape((image.shape[0], image.shape[1])), centers


    def _resize_image(self, image:np.ndarray, new_width:int, src_height:int, src_width:int) -> np.ndarray:
        """縦横比を保って画像をリサイズ"""
        if new_width <= 0:
            raise ValueError("new_width and new_height must be positive integers.")
        
        resize_ratio = new_width / src_width
        new_height = int(src_height * resize_ratio)
        new_size = (new_width, new_height)

        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
        return resized_image


    def _resize_image_2slim(self, image:np.ndarray) -> np.ndarray:
        """画像を縦に引き伸ばす"""
        if self._cell_height <= 0 or self._cell_width <= 0:
            raise ValueError("cell_height and cell_width must be positive integers.")
        
        height, width = image.shape[:2]
        new_size = (int(width), int((height * self._cell_width) / self._cell_height))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        return resized_image


    def _image_to_pixelize(self, image:np.ndarray) -> np.ndarray:
        """画像のピクセル間の線を引く"""
        height, width = image.shape[:2]

        if self._thick_line_interval <= 0:
            raise ValueError("thick_line_interval must be a positive integer.")

        out_height = height * self._cell_height + (height // self._thick_line_interval) * (self._thick_line_thickness - self._line_thickness) + (height - 1) * self._line_thickness
        out_width = width * self._cell_width + (width // self._thick_line_interval) * (self._thick_line_thickness - self._line_thickness) + (width - 1) * self._line_thickness
        output_image = np.full((out_height, out_width, 3), self._background_color, dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                start_y = i * (self._cell_height + self._line_thickness) + (i // self._thick_line_interval) * (self._thick_line_thickness - self._line_thickness)
                start_x = j * (self._cell_width + self._line_thickness) + (j // self._thick_line_interval) * (self._thick_line_thickness - self._line_thickness)
                output_image[start_y:start_y + self._cell_height, start_x:start_x + self._cell_width] = image[i, j]

        return output_image

    
    # ==================== Public Methods ====================
    def create_label_image(self, src: np.ndarray)-> tuple[np.ndarray, list[Color]]:
        """画像からcentersのインデックスの1channel画像を作成"""

        if max(src.shape) > 2048:
            scale = 2048 / max(src.shape)
            new_size = (int(src.shape[1] * scale), int(src.shape[0] * scale))
            src = cv2.resize(src, new_size, interpolation=cv2.INTER_CUBIC)
        
        resize = self._resize_image_2slim(src)
        median, labels, centers = self._median_cut(resize)
        
        labels = labels.astype(np.uint8)
        noise = self._remove_noise_ori(median)

        dst = self._resize_image(noise, self._number_of_line_cells, noise.shape[0], noise.shape[1])
        label_image = _map_image_to_center_color(dst, centers)

        if self._denoise:
            label_image = self._remove_noise(label_image)

        palette = _load_color_csv("data/merinorainbow.csv")
        mapped_colors = _map_colors_to_palette(centers, palette)

        return label_image, mapped_colors


    def create_pixel_image(self, label_image: np.ndarray, mapped_colors: list[Color])-> np.ndarray:
        """label_imageのインデックスをmapped_colorsのRGBに変換して、ピクセル化する"""
        ret = _map_image_colors_to_colors(label_image, mapped_colors)
        pixel = self._image_to_pixelize(ret)
        return pixel


    def create_mapped_image(self, label_image: np.ndarray, mapped_colors: list[Color])-> np.ndarray:
        """label_imageをmapped_colorsのBGR画像に変換する（グリッド線なし）"""
        return _map_image_colors_to_colors(label_image, mapped_colors)


    def create_color_counts(self, label_image: np.ndarray, mapped_colors: list[Color])-> list[ColorCount]:
        """label_imageのインデックスをmapped_colorsのRGBに変換して、色ごとのピクセル数をカウントする"""
        color_counts = _count_color_pixels(label_image)
        color_counts = [ColorCount(color.type, color.color_number, color.rgb, color.lab, count)
                        for color, count in zip(mapped_colors, color_counts)]
        color_counts = sorted(color_counts, key=lambda c: c.count, reverse=True)
        return color_counts


    def run(self, filename: str|None = None, src: np.ndarray|None = None)-> tuple[np.ndarray, list[ColorCount]]:
        if filename is not None:
            src = cv2.imread(filename)
            if src is None:
                raise ValueError("Image not found or unable to load.")
        elif src is None:
            raise ValueError("Either filename or src must be provided.")
        
        label_image, mapped_colors = self.create_label_image(src)
        created_pixel_image = self.create_pixel_image(label_image, mapped_colors)
        color_counts = self.create_color_counts(label_image, mapped_colors)
        return created_pixel_image, color_counts


# ==================== Static/Utility Functions ====================
def display_image(image):
    """引数で画像ファイルを受け取り、画像を表示する関数"""
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# csv読み込み
# 系統,色番,R,G,B,コメント
# 一行目はヘッダー
def _load_color_csv(file_path: str) -> list:
    colors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            system, color_number, r, g, b, _ = parts
            rgb = [int(r), int(g), int(b)]

            # L*a*b*に変換
            rgb_mat = np.array([[rgb]], dtype=np.uint8)
            lab = cv2.cvtColor(rgb_mat, cv2.COLOR_RGB2LAB)[0][0].tolist()

            color = Color(system, color_number, rgb, lab)
            colors.append(color)
    return colors


def _nearest_color(target_rgb, color_list:List[Color]) -> Color:
    rgb_mat = np.array([[target_rgb]], dtype=np.uint8)
    target_lab = cv2.cvtColor(rgb_mat, cv2.COLOR_RGB2LAB)[0][0].tolist()
    
    min_distance = float('inf')
    nearest = None
    for color in color_list:
        l, a, b = color.lab  # L*a*b*で距離を測る
        distance = ((float(target_lab[0]) - float(l)) * 100 / 255) ** 2
        distance += (float(target_lab[1]) - float(a)) ** 2
        distance += (float(target_lab[2]) - float(b)) ** 2
        if distance < min_distance:
            min_distance = distance
            nearest = color
    if nearest is None:
        raise ValueError("No nearest color found. Check if color_list is empty.")
    return nearest


def _map_colors_to_palette(centers, palette)->List[Color]:
    mapped_colors = []
    for center in centers:
        nearest = _nearest_color(list(reversed(center)), palette)
        mapped_colors.append(nearest)
    return mapped_colors


def _count_color_pixels(grey)->List[int]:
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
        if idx >= len(ret):
            continue
        ret[idx] = count
    return ret


def _map_image_to_center_color(image, centers)->np.ndarray:
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
    return grey


# label_imageのインデックスをmapped_colorsのRGBに変換
def _map_image_colors_to_colors(label_image, mapped_colors)->np.ndarray:
    height, width = label_image.shape[:2]
    mapped_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            idx = label_image[i, j]
            if idx < len(mapped_colors):
                mapped_image[i, j] = list(reversed(mapped_colors[idx].rgb))
            else:
                mapped_image[i, j] = [255, 255, 255]  # centersにない色は白にする
    return mapped_image


# ==================== Main ====================
if __name__ == "__main__":
    # 第一引数から画像パス取得
    if len(sys.argv) < 2:
        raise ValueError("Usage: python image2cells.py <image_path>")
    file_path = sys.argv[1]

    processor = ImageToPixels()
    pixels, color_counts = processor.run(file_path)
    
    cv2.imwrite("output_pixelized.png", pixels)
    # print("Centers:", centers)
    print("Color Counts:", color_counts)

    display_image(pixels)