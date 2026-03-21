import os
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import io
import hashlib
import uuid
import pandas as pd

# srcフォルダをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image2cells import ImageToPixels
from streamlit_image_coordinates import streamlit_image_coordinates

def resource_path(*parts: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base_dir.joinpath(*parts)

# 例: 固定CSVの参照
MERINO_RAINBOW_CSV = resource_path("data", "merinorainbow.csv")

# ==================== 定数定義 ====================
# UI設定
DEFAULT_COLORS_NUMBER = 12
DEFAULT_LINE_CELLS = 64
DEFAULT_CELL_HEIGHT = 27
DEFAULT_CELL_WIDTH = 33
DEFAULT_LINE_THICKNESS = 1
DEFAULT_THICK_LINE_THICKNESS = 3
DEFAULT_THICK_LINE_INTERVAL = 5
DEFAULT_EDIT_SCALE = 10
DEFAULT_EDIT_HISTORY_LIMIT = 500

# スライダー範囲
COLORS_NUMBER_RANGE = (2, 64)
LINE_CELLS_RANGE = (16, 256)
CELL_HEIGHT_RANGE = (10, 100)
CELL_WIDTH_RANGE = (10, 100)
LINE_THICKNESS_RANGE = (1, 10)
THICK_LINE_THICKNESS_RANGE = (2, 10)
THICK_LINE_INTERVAL_RANGE = (1, 20)
EDIT_SCALE_RANGE = (4, 20)

# 画像表示設定
MAX_DISPLAY_WIDTH = 800
MAX_ROI_DISPLAY_WIDTH = 560
MAX_EDIT_DISPLAY_WIDTH = 1200
CIRCLE_RADIUS = 8
CIRCLE_THICKNESS = -1
RECTANGLE_THICKNESS = 3
OVERLAY_ALPHA = 0.15
TEXT_BRIGHTNESS_THRESHOLD = 150

# ファイル設定
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def _ensure_tmp_dir():
    Path("./tmp").mkdir(parents=True, exist_ok=True)


# ==================== ヘルパー関数 ====================
def get_rgb_color_html(rgb_tuple:tuple[int, int, int]) -> str:
    """RGB値をHTMLカラーコード形式に変換"""
    return f"rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})"


def draw_color_sample(rgb_tuple:tuple[int, int, int], width:int=40, height:int=40) -> str:
    """色見本HTMLを生成"""
    rgb_color = get_rgb_color_html(rgb_tuple)
    return f'<div style="width: {width}px; height: {height}px; background-color: {rgb_color}; border: 1px solid #ccc; border-radius: 4px;"></div>'


def get_contrast_text_color(bgr_tuple: tuple[int, int, int]) -> tuple[int, int, int]:
    """背景色に応じて視認性の高い文字色（黒 or 白）を返す"""
    b, g, r = bgr_tuple
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if brightness >= TEXT_BRIGHTNESS_THRESHOLD else (255, 255, 255)


def build_color_code_grid(label_image: np.ndarray, mapped_colors: list) -> np.ndarray:
    """セル(y, x)ごとの色コード文字列を格納した (height, width) の配列を返す"""
    codes = np.array([str(c.color_number) for c in mapped_colors], dtype=object)
    safe_idx = np.clip(label_image.astype(np.int32), 0, len(mapped_colors) - 1)
    return codes[safe_idx]


def create_color_code_csv(color_code_grid: np.ndarray) -> bytes:
    """color_code_gridからCSVを生成する（UTF-8・BOMなしのバイト列を返す）"""
    # 画像と同じ配置（行=y、列=x）のままCSV化
    df = pd.DataFrame(color_code_grid)
    csv_str = df.to_csv(index=False, header=False)
    return csv_str.encode("utf-8")


def build_color_code_cache_key(
    label_image: np.ndarray,
    mapped_colors: list,
    processor: ImageToPixels,
) -> str:
    """色コード画像/CSVのキャッシュキーを生成する"""
    hasher = hashlib.blake2b(digest_size=16)
    # label_image のバイト列だけでなく shape / dtype もキーに含めて衝突を防ぐ
    hasher.update(str((label_image.shape, str(label_image.dtype))).encode("utf-8"))
    hasher.update(label_image.tobytes())

    for color in mapped_colors:
        hasher.update(str(color.color_number).encode("utf-8"))
        hasher.update(bytes(color.rgb))

    config_tuple = (
        processor.cell_height,
        processor.cell_width,
        processor.line_thickness,
        processor.thick_line_thickness,
        processor.thick_line_interval,
    )
    hasher.update(str(config_tuple).encode("utf-8"))
    return hasher.hexdigest()


def create_color_code_pixel_image(
    label_image: np.ndarray,
    mapped_colors: list,
    processor: ImageToPixels,
    color_code_grid: np.ndarray,
    base_pixel: np.ndarray | None = None,
) -> np.ndarray:
    """各セル中央に色コードを重ねたドット絵画像を作成する"""
    coded_pixel = base_pixel.copy() if base_pixel is not None else processor.create_pixel_image(label_image, mapped_colors)
    height, width = label_image.shape[:2]
    cell_h = processor.cell_height
    cell_w = processor.cell_width
    line_thickness = processor.line_thickness
    thick_line_interval = processor.thick_line_interval
    thick_line_delta = processor.thick_line_thickness - line_thickness

    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = max(0.25, min(cell_w, cell_h) / 42.0)

    row_starts = [
        y * (cell_h + line_thickness) + (y // thick_line_interval) * thick_line_delta
        for y in range(height)
    ]
    col_starts = [
        x * (cell_w + line_thickness) + (x // thick_line_interval) * thick_line_delta
        for x in range(width)
    ]

    # 描画パラメータのみをキャッシュ（テキスト文字列は color_code_grid から取得）
    render_cache = {}
    for idx, color in enumerate(mapped_colors):
        color_code = str(color.color_number).strip()
        if not color_code:
            render_cache[idx] = None
            continue

        font_scale = base_font_scale
        thickness = max(1, int(round(font_scale)))
        text_size, _ = cv2.getTextSize(color_code, font, font_scale, thickness)
        text_w, text_h = text_size

        while (text_w > cell_w - 2 or text_h > cell_h - 2) and font_scale > 0.2:
            font_scale *= 0.9
            thickness = max(1, int(round(font_scale)))
            text_size, _ = cv2.getTextSize(color_code, font, font_scale, thickness)
            text_w, text_h = text_size

        cell_bgr = tuple(int(v) for v in reversed(color.rgb))
        render_cache[idx] = {
            "font_scale": font_scale,
            "thickness": thickness,
            "offset_x": max(0, (cell_w - text_w) // 2),
            "offset_y": max(text_h, (cell_h + text_h) // 2),
            "text_color": get_contrast_text_color(cell_bgr),
        }

    for y in range(height):
        start_y = row_starts[y]
        for x in range(width):
            color_idx = int(label_image[y, x])
            if color_idx >= len(mapped_colors):
                continue

            params = render_cache.get(color_idx)
            if params is None:
                continue

            text = color_code_grid[y, x]
            if not text:
                continue

            start_x = col_starts[x]
            cv2.putText(
                coded_pixel,
                text,
                (start_x + params["offset_x"], start_y + params["offset_y"]),
                font,
                params["font_scale"],
                params["text_color"],
                params["thickness"],
                cv2.LINE_AA,
            )

    return coded_pixel


def resize_for_display(image:np.ndarray, max_width:int=MAX_DISPLAY_WIDTH) -> tuple[np.ndarray, float]:
    """画像を表示用にリサイズし、表示画像と縮小率を返す"""
    orig_height, orig_width = image.shape[:2]
    if orig_width > max_width:
        display_scale = max_width / orig_width
        display_width = max_width
        display_height = int(orig_height * display_scale)
        display_resized = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        return display_resized, display_scale
    return image, 1.0


def create_colors_csv(mapped_colors:list) -> str:
    """色情報をCSV形式で生成"""
    colors_data = []
    for idx, color in enumerate(mapped_colors):
        colors_data.append({
            "色番": idx,
            "色名": color.type,
            "色コード": color.color_number,
            "Amazonリンク": color.amazon_url,
            "R": color.rgb[0],
            "G": color.rgb[1],
            "B": color.rgb[2]
        })
    colors_df = pd.DataFrame(colors_data)
    return colors_df.to_csv(index=False)


def get_rect_dimensions(rect:tuple[int, int, int, int]) -> tuple[int, int]:
    """矩形情報から幅と高さを取得"""
    x1, y1, x2, y2 = rect
    return x2 - x1, y2 - y1


def get_effective_roi_dimensions(src_shape: tuple[int, int, int]) -> tuple[int, int, str]:
    """現在のROIから処理対象の幅・高さと表示ラベルを返す"""
    src_height, src_width = src_shape[:2]
    roi_rect = st.session_state.get("roi_rect")

    if not roi_rect:
        return src_width, src_height, "画像全体"

    x1, y1, x2, y2 = roi_rect
    width = max(1, min(src_width, x2) - max(0, x1))
    height = max(1, min(src_height, y2) - max(0, y1))

    is_full = x1 <= 0 and y1 <= 0 and x2 >= src_width and y2 >= src_height
    label = "画像全体" if is_full else "選択矩形"
    return width, height, label


def estimate_vertical_cells(
    target_width: int,
    target_height: int,
    horizontal_cells: int,
    cell_width: int,
    cell_height: int,
) -> int:
    """変換ロジックに合わせて縦セル数を推定"""
    slim_height = target_height * (cell_width / cell_height)
    vertical_cells = int(slim_height * (horizontal_cells / target_width))
    return max(1, vertical_cells)


def format_color_option(color_idx: int, mapped_colors: list) -> str:
    """色選択UI用の表示名を返す"""
    color = mapped_colors[color_idx]
    return f"{color.type} ({color.color_number})"


def build_edit_operation(coords: np.ndarray, prev_idx: int, new_idx: int, op_type: str) -> dict:
    """Undo/Redo 用の編集操作を構築"""
    return {
        "type": op_type,
        "coords": np.asarray(coords, dtype=np.int32),
        "prev": int(prev_idx),
        "new": int(new_idx),
        "count": int(len(coords)),
    }


def apply_edit_operation(operation: dict, use_new_value: bool):
    """編集操作を label_image / mapped_image に反映する"""
    if "coords" in operation:
        coords = np.asarray(operation["coords"], dtype=np.int32)
        if coords.size == 0:
            return
        target_idx = int(operation["new"] if use_new_value else operation["prev"])
        ys = coords[:, 0]
        xs = coords[:, 1]
        st.session_state.label_image[ys, xs] = target_idx
        target_bgr = tuple(reversed(st.session_state.mapped_colors[target_idx].rgb))
        st.session_state.mapped_image[ys, xs] = target_bgr
        return

    if "changes" in operation:
        for change in operation["changes"]:
            target_idx = int(change["new"] if use_new_value else change["prev"])
            x = int(change["x"])
            y = int(change["y"])
            st.session_state.label_image[y, x] = target_idx
            target_bgr = tuple(reversed(st.session_state.mapped_colors[target_idx].rgb))
            st.session_state.mapped_image[y, x] = target_bgr
        return

    if all(key in operation for key in ("x", "y", "prev", "new")):
        target_idx = int(operation["new"] if use_new_value else operation["prev"])
        x = int(operation["x"])
        y = int(operation["y"])
        st.session_state.label_image[y, x] = target_idx
        target_bgr = tuple(reversed(st.session_state.mapped_colors[target_idx].rgb))
        st.session_state.mapped_image[y, x] = target_bgr


def push_edit_operation(operation: dict):
    """編集履歴へ操作を追加"""
    st.session_state.edit_history.append(operation)
    if len(st.session_state.edit_history) > DEFAULT_EDIT_HISTORY_LIMIT:
        st.session_state.edit_history.pop(0)
    st.session_state.redo_history.clear()


def set_replace_color_from_selected(target_key: str):
    """現在の選択色を置換元/置換先へ反映"""
    st.session_state[target_key] = st.session_state.selected_color_idx


def init_session_state(src_image:np.ndarray):
    """セッション状態の初期化"""
    if "roi_p1" not in st.session_state:
        st.session_state.roi_p1 = None
        st.session_state.roi_p2 = None
        st.session_state.roi_selecting_point = None  # None, "p1", "p2" の3値
        st.session_state.last_click_coords = None  # 前回のクリック座標
        # デフォルト：画像全体
        height, width = src_image.shape[:2]
        st.session_state.roi_rect = (0, 0, width, height)


def setup_sidebar():
    """サイドバーの設定を行い、パラメータを返す"""
    st.sidebar.header("⚙️ Step 4: 変換設定")
    st.sidebar.caption("まずは基本設定で処理し、必要なら詳細設定を調整してください。")
    
    colors_number = st.sidebar.slider(
        "使用する色数",
        min_value=COLORS_NUMBER_RANGE[0],
        max_value=COLORS_NUMBER_RANGE[1],
        value=DEFAULT_COLORS_NUMBER,
        step=1
    )
    number_of_line_cells = st.sidebar.slider(
        "横セル数",
        min_value=LINE_CELLS_RANGE[0],
        max_value=LINE_CELLS_RANGE[1],
        value=DEFAULT_LINE_CELLS,
        step=8
    )
    denoise = st.sidebar.checkbox("ノイズ除去を有効にする", value=False)

    with st.sidebar.expander("詳細設定", expanded=False):
        cell_height = st.slider(
            "セル高さ",
            min_value=CELL_HEIGHT_RANGE[0],
            max_value=CELL_HEIGHT_RANGE[1],
            value=DEFAULT_CELL_HEIGHT,
            step=1
        )
        cell_width = st.slider(
            "セル幅",
            min_value=CELL_WIDTH_RANGE[0],
            max_value=CELL_WIDTH_RANGE[1],
            value=DEFAULT_CELL_WIDTH,
            step=1
        )
        line_thickness = st.slider(
            "通常グリッド線の太さ",
            min_value=LINE_THICKNESS_RANGE[0],
            max_value=LINE_THICKNESS_RANGE[1],
            value=DEFAULT_LINE_THICKNESS,
            step=1
        )
        thick_line_thickness = st.slider(
            "太いグリッド線の太さ",
            min_value=THICK_LINE_THICKNESS_RANGE[0],
            max_value=THICK_LINE_THICKNESS_RANGE[1],
            value=DEFAULT_THICK_LINE_THICKNESS,
            step=1
        )
        thick_line_interval = st.slider(
            "太いグリッド線の間隔（セル数）",
            min_value=THICK_LINE_INTERVAL_RANGE[0],
            max_value=THICK_LINE_INTERVAL_RANGE[1],
            value=DEFAULT_THICK_LINE_INTERVAL,
            step=1
        )

    return {
        "colors_number": colors_number,
        "number_of_line_cells": number_of_line_cells,
        "cell_height": cell_height,
        "cell_width": cell_width,
        "line_thickness": line_thickness,
        "thick_line_thickness": thick_line_thickness,
        "thick_line_interval": thick_line_interval,
        "denoise": denoise
    }


def render_exit_button():
    """サイドバー最下部に終了ボタンを表示"""
    st.sidebar.markdown("---")
    if st.sidebar.button("アプリを終了", use_container_width=True, key="exit_app_button"):
        os._exit(0)


def upload_image_section():
    """画像アップロードUI"""
    st.sidebar.header("📁 Step 1: 画像を選択")
    uploaded_file = st.sidebar.file_uploader(
        "画像ファイルを選択してください",
        type=SUPPORTED_IMAGE_FORMATS
    )
    return uploaded_file


def upload_csv_section() -> bytes:
    """毛糸CSVアップロードUI。使用するCSVのバイト列を返す"""
    st.sidebar.header("🧶 Step 2: 毛糸CSVを選択（任意）")
    st.sidebar.caption(
        "デフォルトはメリノレインボーCSVを使用します。"
        "別の毛糸を使う場合はCSVをアップロードしてください。"
    )
    uploaded_csv = st.sidebar.file_uploader(
        "毛糸CSVファイル（任意）",
        type=["csv"],
        key="csv_uploader"
    )

    if uploaded_csv is not None:
        # アップロードされたファイルをバイト列で返す
        csv_bytes = uploaded_csv.read()
        st.session_state.uploaded_csv_bytes = csv_bytes
        st.session_state.uploaded_csv_name = uploaded_csv.name
        st.sidebar.success(f"📄 使用中: {uploaded_csv.name}")
        return csv_bytes
    else:
        # デフォルトCSVをファイルから読み込んでバイト列で返す
        csv_bytes = Path(MERINO_RAINBOW_CSV).read_bytes()
        st.sidebar.caption("📄 使用中: merinorainbow.csv（デフォルト）")
        return csv_bytes


def render_roi_selection_ui(src_shape: tuple[int, int, int], display_image:np.ndarray, display_scale:float):
    """ROI選択UIを描画し、クリック座標を処理"""
    with st.expander("🔲 画像内の領域を選択（オプション）", expanded=False):
        st.caption("デフォルトでは画像全体を処理します。特定の領域のみを処理したい場合に設定してください。")
        
        # 選択状態の表示
        p1_status = "✅" if st.session_state.roi_p1 else "⭕"
        p2_status = "✅" if st.session_state.roi_p2 else "⭕"
        
        select_col1, select_col2, select_col3 = st.columns(3)
        
        with select_col1:
            button_text = f"📍 左上 {p1_status}"
            if st.button(button_text, use_container_width=True, type="secondary", key="btn_p1"):
                st.session_state.roi_selecting_point = "p1"
                st.session_state.last_click_coords = None
                st.rerun()
        
        with select_col2:
            button_text = f"📍 右下 {p2_status}"
            if st.button(button_text, use_container_width=True, type="secondary", key="btn_p2"):
                st.session_state.roi_selecting_point = "p2"
                st.session_state.last_click_coords = None
                st.rerun()
        
        with select_col3:
            if st.button("🔄 リセット", use_container_width=True, key="btn_reset"):
                st.session_state.roi_p1 = None
                st.session_state.roi_p2 = None
                st.session_state.last_click_coords = None
                st.session_state.roi_selecting_point = None
                height, width = src_shape[:2]
                st.session_state.roi_rect = (0, 0, width, height)
        
        st.markdown(
            f"**選択状態**: 左上 {p1_status} `{st.session_state.roi_p1 if st.session_state.roi_p1 else '未選択'}` | "
            f"右下 {p2_status} `{st.session_state.roi_p2 if st.session_state.roi_p2 else '未選択'}`"
        )
    
    # 画像表示とインタラクション
    display_image_copy = display_image.copy()
    
    # 選択済みポイントを描画（リサイズスケールを考慮）
    if st.session_state.roi_p1:
        p1_scaled = (int(st.session_state.roi_p1[0] * display_scale), int(st.session_state.roi_p1[1] * display_scale))
        cv2.circle(display_image_copy, p1_scaled, CIRCLE_RADIUS, (0, 255, 0), CIRCLE_THICKNESS)
        cv2.putText(display_image_copy, "P1(LT)", (p1_scaled[0] + 10, p1_scaled[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if st.session_state.roi_p2:
        p2_scaled = (int(st.session_state.roi_p2[0] * display_scale), int(st.session_state.roi_p2[1] * display_scale))
        cv2.circle(display_image_copy, p2_scaled, CIRCLE_RADIUS, (255, 0, 0), CIRCLE_THICKNESS)
        cv2.putText(display_image_copy, "P2(RB)", (p2_scaled[0] + 10, p2_scaled[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 両点が選択されたら矩形を描画（リサイズスケールを考慮）
    if st.session_state.roi_p1 and st.session_state.roi_p2:
        p1 = st.session_state.roi_p1
        p2 = st.session_state.roi_p2
        x1, x2 = sorted([p1[0], p2[0]])
        y1, y2 = sorted([p1[1], p2[1]])
        # リサイズスケールを適用
        x1_scaled = int(x1 * display_scale)
        y1_scaled = int(y1 * display_scale)
        x2_scaled = int(x2 * display_scale)
        y2_scaled = int(y2 * display_scale)
        cv2.rectangle(display_image_copy, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), RECTANGLE_THICKNESS)
        # 矩形内を半透明に
        overlay = display_image_copy.copy()
        cv2.rectangle(overlay, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), -1)
        display_image_copy = cv2.addWeighted(overlay, OVERLAY_ALPHA, display_image_copy, 1 - OVERLAY_ALPHA, 0)
    
    # 画像をクリック可能にして座標取得
    coords = streamlit_image_coordinates(
        source=cv2.cvtColor(display_image_copy, cv2.COLOR_BGR2RGB), key="roi_selector")
    
    if coords is not None and "x" in coords and "y" in coords:
        # クリック座標を元の画像サイズに変換
        click_point = (int(coords["x"] / display_scale), int(coords["y"] / display_scale))
        
        # 前回とは異なるクリックかどうかを確認
        if click_point != st.session_state.last_click_coords:
            st.session_state.last_click_coords = click_point
            
            # roi_selecting_point が設定されている場合のみ座標を保存
            if st.session_state.roi_selecting_point == "p1":
                st.session_state.roi_p1 = click_point
                st.session_state.roi_selecting_point = None
                st.success(f"✅ 左上: {click_point}")
            elif st.session_state.roi_selecting_point == "p2":
                st.session_state.roi_p2 = click_point
                st.session_state.roi_selecting_point = None
                st.success(f"✅ 右下: {click_point}")
            
            # 両点が選択されたら矩形を確定
            if st.session_state.roi_p1 and st.session_state.roi_p2:
                p1 = st.session_state.roi_p1
                p2 = st.session_state.roi_p2
                x1, x2 = sorted([p1[0], p2[0]])
                y1, y2 = sorted([p1[1], p2[1]])
                st.session_state.roi_rect = (x1, y1, x2, y2)
            
            st.rerun()


def process_selected_roi(src_image:np.ndarray, process_image:np.ndarray) -> np.ndarray|None:
    """ROIに基づいて画像を処理"""
    if st.session_state.roi_rect:
        x1, y1, x2, y2 = st.session_state.roi_rect
        
        # 画像全体かどうかをチェック
        is_full_image = (x1 == 0 and y1 == 0 and 
                       x2 == src_image.shape[1] and y2 == src_image.shape[0])
        
        if not is_full_image:
            # 矩形サイズの検証
            if x1 >= x2 or y1 >= y2:
                st.error(f"⚠️ 矩形のサイズが不正です: ({x1}, {y1}) - ({x2}, {y2})")
                return None
            elif (x2 - x1) < 2 or (y2 - y1) < 2:
                st.error(f"⚠️ 矩形が小さすぎます: 幅{x2-x1}px, 高さ{y2-y1}px（最小2px必要）")
                return None
            else:
                process_image = src_image[y1:y2, x1:x2].copy()
                st.session_state.roi_offset = (x1, y1)
                
                if process_image.size == 0:
                    st.error("⚠️ 抽出した画像が空です")
                    return None
                
                st.info(f"📍 処理対象: 選択領域 位置({x1}, {y1}) サイズ {x2-x1}×{y2-y1}")
        else:
            st.session_state.roi_offset = (0, 0)
            st.info("📍 処理対象: 画像全体")
    else:
        st.session_state.roi_offset = (0, 0)
    
    return process_image


def render_result_image():
    """処理結果画像を表示（col2 内で使用）"""
    st.subheader("処理結果")
    st.image(st.session_state.result_pixel, use_container_width=True, channels="BGR")


def render_details_section(src_image:np.ndarray):
    """詳細情報セクション"""
    st.markdown("---")
    st.subheader("📊  詳細情報")
    
    # 矩形選択情報の表示（選択されている場合のみ）
    if "roi_rect" in st.session_state and st.session_state.roi_rect:
        x1, y1, x2, y2 = st.session_state.roi_rect
        roi_width, roi_height = get_rect_dimensions(st.session_state.roi_rect)
        # 画像全体かどうかをチェック
        if (x1, y1) != (0, 0) or (roi_width, roi_height) != src_image.shape[:2][::-1]:
            st.info(f"📍 選択領域: 位置({x1}, {y1}) サイズ {roi_width}×{roi_height}")
        else:
            st.info("📍 処理対象: 画像全体")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.metric("取得した色数", len(st.session_state.color_counts))
    
    with info_col2:
        st.metric("ピクセル総数", sum(c.count for c in st.session_state.color_counts))
    
    # 色カウント情報
    st.markdown("#### 🎨 色ごとのピクセル数")
    
    # 各色の情報を表示
    for idx, color in enumerate(st.session_state.color_counts):
        col1, col2, col3, col4 = st.columns([0.1, 0.4, 0.3, 0.3])
        
        with col1:
            st.markdown(draw_color_sample(color.rgb), unsafe_allow_html=True)
        
        with col2:
            st.text(f"**{color.type}** ({color.color_number})")
        
        with col3:
            st.metric("ピクセル数", f"{color.count:,}", label_visibility="collapsed")

        with col4:
            if color.amazon_url:
                st.link_button("Amazon", color.amazon_url, use_container_width=True)
            else:
                st.caption("リンクなし")
    
    # 処理結果のダウンロード
    st.markdown("---")
    st.subheader("📥 結果をダウンロード")
    
    colors_csv = create_colors_csv(st.session_state.mapped_colors)
    
    # ダウンロード用データ生成
    _, img_bytes = cv2.imencode('.png', st.session_state.result_pixel)
    img_buffer = io.BytesIO(img_bytes)
    processor = st.session_state.get("processor", ImageToPixels())
    color_code_cache_key = build_color_code_cache_key(
        st.session_state.label_image,
        st.session_state.mapped_colors,
        processor,
    )

    if st.session_state.get("color_code_cache_key") != color_code_cache_key:
        color_code_grid = build_color_code_grid(
            st.session_state.label_image,
            st.session_state.mapped_colors,
        )
        coded_pixel = create_color_code_pixel_image(
            st.session_state.label_image,
            st.session_state.mapped_colors,
            processor,
            color_code_grid,
            base_pixel=st.session_state.result_pixel,
        )
        _, coded_img_bytes = cv2.imencode('.png', coded_pixel)
        st.session_state.color_code_cache_key = color_code_cache_key
        st.session_state.cached_color_code_png = io.BytesIO(coded_img_bytes)
        st.session_state.cached_color_code_csv = create_color_code_csv(color_code_grid)

    coded_img_buffer = st.session_state.cached_color_code_png
    color_code_csv = st.session_state.cached_color_code_csv

    # ダウンロードボタン
    st.markdown("#### 🎯 基本用途（通常はこれだけで十分です）")
    col_code_img, col_csv = st.columns(2)
    
    with col_code_img:
        st.markdown("**色コード付きドット絵**")
        st.caption("各セルに色番号が書かれた図。編み物をしながら参照するのに最適です。")
        st.download_button(
            label="🔢 ダウンロード",
            data=coded_img_buffer,
            file_name="result_pixelized_with_color_code.png",
            mime="image/png",
            use_container_width=True,
            key="dl_coded_pixel"
        )

    with col_csv:
        st.markdown("**毛糸の色情報**")
        st.caption("使用する毛糸の色名と商品リンク。毛糸購入時に使います。")
        st.download_button(
            label="🛒 ダウンロード",
            data=colors_csv,
            file_name="color_palette.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_colors_csv"
        )

    with st.expander("📦 詳細用途（必要に応じて）", expanded=False):
        st.caption("下記は特定の用途に使用します。通常は不要です。")
        col_img, col_code_csv = st.columns(2)
        
        with col_img:
            st.markdown("**通常のドット絵（グリッド線のみ）**")
            st.caption("色番号なし。シンプルな図が必要な場合に使用します。")
            st.download_button(
                label="🖼️ ダウンロード",
                data=img_buffer,
                file_name="result_pixelized.png",
                mime="image/png",
                use_container_width=True,
                key="dl_plain_pixel"
            )
        
        with col_code_csv:
            st.markdown("**色コード配列（エクセル用）**")
            st.caption("各セルの色番号をCSV形式で。スプレッドシートで分析・加工する場合に使用します。")
            st.download_button(
                label="📋 ダウンロード",
                data=color_code_csv,
                file_name="color_code_map.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_code_csv"
            )


def render_edit_section():
    """編集UI"""
    st.markdown("---")
    st.subheader("🖌️ 編集")
    
    if streamlit_image_coordinates is None:
        st.warning("編集UIを使うには streamlit-image-coordinates の導入が必要です。")
        return
    
    edit_scale = st.sidebar.slider(
        "編集表示倍率",
        min_value=EDIT_SCALE_RANGE[0],
        max_value=EDIT_SCALE_RANGE[1],
        value=DEFAULT_EDIT_SCALE,
        step=1
    )
    
    if "selected_color_idx" not in st.session_state:
        st.session_state.selected_color_idx = 0
    if st.session_state.selected_color_idx >= len(st.session_state.mapped_colors):
        st.session_state.selected_color_idx = 0
    if "editor_mode" not in st.session_state:
        st.session_state.editor_mode = "塗る"
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = []
    if "redo_history" not in st.session_state:
        st.session_state.redo_history = []
    if "replace_source_idx" not in st.session_state:
        st.session_state.replace_source_idx = 0
    if "replace_target_idx" not in st.session_state:
        st.session_state.replace_target_idx = 1 if len(st.session_state.mapped_colors) > 1 else 0

    max_color_index = len(st.session_state.mapped_colors) - 1
    st.session_state.replace_source_idx = min(st.session_state.replace_source_idx, max_color_index)
    st.session_state.replace_target_idx = min(st.session_state.replace_target_idx, max_color_index)

    notice = st.session_state.pop("edit_notice", None) if "edit_notice" in st.session_state else None
    if notice:
        notice_type = notice.get("type", "info")
        notice_text = notice.get("text", "")
        if notice_type == "success":
            st.success(notice_text)
        elif notice_type == "warning":
            st.warning(notice_text)
        else:
            st.info(notice_text)

    st.markdown("#### 🧰 操作モード")
    editor_mode = st.radio(
        "操作モード",
        ["塗る", "スポイト"],
        key="editor_mode",
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_color = st.session_state.mapped_colors[st.session_state.selected_color_idx]
    selected_col1, selected_col2 = st.columns([0.08, 0.92])
    with selected_col1:
        st.markdown(draw_color_sample(selected_color.rgb, width=24, height=24), unsafe_allow_html=True)
    with selected_col2:
        st.caption(f"選択中色: {selected_color.type} ({selected_color.color_number})")

    if editor_mode == "スポイト":
        st.caption("スポイトモード: 画像をクリックすると、そのセルの色が選択されます。")
    else:
        st.caption("塗るモード: 画像をクリックすると、選択中の色で塗ります。")
    
    label_image = st.session_state.label_image
    processor = st.session_state.get("processor", ImageToPixels())

    if "mapped_image" not in st.session_state:
        st.session_state.mapped_image = processor.create_mapped_image(
            label_image,
            st.session_state.mapped_colors
        )
    if "original_mapped_image" not in st.session_state:
        st.session_state.original_mapped_image = st.session_state.mapped_image.copy()

    mapped_image = st.session_state.mapped_image
    
    height, width = mapped_image.shape[:2]
    preview = cv2.resize(
        mapped_image,
        (width * edit_scale, height * edit_scale),
        interpolation=cv2.INTER_NEAREST
    )
    # 横長画像でもUIからはみ出さないように、表示用だけ横幅を制限
    preview_display, preview_display_scale = resize_for_display(
        preview,
        max_width=MAX_EDIT_DISPLAY_WIDTH
    )
    preview_rgb = cv2.cvtColor(preview_display, cv2.COLOR_BGR2RGB)
    
    st.markdown("#### 🧭 クリック操作")
    coords = streamlit_image_coordinates(preview_rgb, key="editor_canvas")
    if coords is not None and "x" in coords and "y" in coords:
        click = (coords["x"], coords["y"])
        if st.session_state.last_click != click:
            st.session_state.last_click = click
            # 表示時に縮小している分を元スケールへ戻してセル位置を計算
            effective_scale = edit_scale * preview_display_scale
            cell_x = int(coords["x"] // effective_scale)
            cell_y = int(coords["y"] // effective_scale)
            if 0 <= cell_x < width and 0 <= cell_y < height:
                if editor_mode == "スポイト":
                    picked_idx = int(st.session_state.label_image[cell_y, cell_x])
                    if st.session_state.selected_color_idx != picked_idx:
                        st.session_state.selected_color_idx = picked_idx
                        st.rerun()
                else:
                    selected_idx = st.session_state.selected_color_idx
                    prev_idx = int(st.session_state.label_image[cell_y, cell_x])
                    if prev_idx != selected_idx:
                        operation = build_edit_operation(
                            coords=np.array([[cell_y, cell_x]], dtype=np.int32),
                            prev_idx=prev_idx,
                            new_idx=selected_idx,
                            op_type="paint"
                        )
                        apply_edit_operation(operation, use_new_value=True)
                        push_edit_operation(operation)
                        st.rerun()

    st.markdown("#### 🎯 色の選択（サブ）")
    palette_cols = st.columns(6)
    for idx, color in enumerate(st.session_state.mapped_colors):
        with palette_cols[idx % 6]:
            st.markdown(draw_color_sample(color.rgb, width=36, height=36), unsafe_allow_html=True)
            label = "選択中" if idx == st.session_state.selected_color_idx else "選択"
            if st.button(label, key=f"palette_{idx}"):
                st.session_state.selected_color_idx = idx
                st.rerun()

    with st.expander("🔁 色をまとめて置換", expanded=True):
        st.caption("置換元と置換先を選んでから実行します。Undo / Redo の対象です。")

        option_indices = list(range(len(st.session_state.mapped_colors)))

        def format_color_select_option(idx: int) -> str:
            return format_color_option(idx, st.session_state.mapped_colors)

        replace_col1, replace_col_mid, replace_col2 = st.columns([1, 0.2, 1])
        with replace_col1:
            st.markdown("**1. 置換元**")
            source_color = st.session_state.mapped_colors[st.session_state.replace_source_idx]
            st.markdown(draw_color_sample(source_color.rgb, width=32, height=32), unsafe_allow_html=True)
            st.selectbox(
                "置換元の色",
                options=option_indices,
                key="replace_source_idx",
                format_func=format_color_select_option,
                label_visibility="collapsed"
            )
            st.button(
                "選択中の色を反映",
                use_container_width=True,
                key="set_replace_source",
                on_click=set_replace_color_from_selected,
                args=("replace_source_idx",)
            )

        with replace_col_mid:
            st.markdown("### →")

        with replace_col2:
            st.markdown("**2. 置換先**")
            target_color = st.session_state.mapped_colors[st.session_state.replace_target_idx]
            st.markdown(draw_color_sample(target_color.rgb, width=32, height=32), unsafe_allow_html=True)
            st.selectbox(
                "置換先の色",
                options=option_indices,
                key="replace_target_idx",
                format_func=format_color_select_option,
                label_visibility="collapsed"
            )
            st.button(
                "選択中の色を反映",
                use_container_width=True,
                key="set_replace_target",
                on_click=set_replace_color_from_selected,
                args=("replace_target_idx",)
            )

        replace_source_idx = st.session_state.replace_source_idx
        replace_target_idx = st.session_state.replace_target_idx
        replace_count = int(np.count_nonzero(st.session_state.label_image == replace_source_idx))

        st.markdown("**3. 実行**")
        if replace_source_idx == replace_target_idx:
            st.info("置換元と置換先が同じです。別の色を選んでください。")
        elif replace_count == 0:
            st.info("現在の画像には置換元のセルがありません。")
        else:
            st.caption(
                f"{format_color_option(replace_source_idx, st.session_state.mapped_colors)} を "
                f"{format_color_option(replace_target_idx, st.session_state.mapped_colors)} に "
                f"{replace_count}セル置換します。"
            )
            if st.button("🔁 この内容で一括置換する（Undo可）", type="secondary", use_container_width=True, key="replace_colors_button"):
                coords = np.argwhere(st.session_state.label_image == replace_source_idx)
                operation = build_edit_operation(
                    coords=coords,
                    prev_idx=replace_source_idx,
                    new_idx=replace_target_idx,
                    op_type="replace"
                )
                apply_edit_operation(operation, use_new_value=True)
                push_edit_operation(operation)
                st.session_state.edit_notice = {
                    "type": "success",
                    "text": (
                        f"{format_color_option(replace_source_idx, st.session_state.mapped_colors)} → "
                        f"{format_color_option(replace_target_idx, st.session_state.mapped_colors)} を "
                        f"{replace_count}セル置換しました。Undo で戻せます。"
                    )
                }
                st.rerun()

    has_pending_edits = len(st.session_state.edit_history) > 0
    if has_pending_edits:
        st.warning("未反映の編集があります。必要なら「✅ 編集内容を結果に反映」を押してください。")
        st.caption(
            f"未反映の編集数: {len(st.session_state.edit_history)} / 履歴上限: {DEFAULT_EDIT_HISTORY_LIMIT}"
        )
        st.caption("Undo / Redo は単セル編集と一括置換の両方に効きます。")
    
    action_col1, action_col2, action_col3 = st.columns([0.2, 0.2, 0.3])
    with action_col1:
        if st.button("↶ Undo", use_container_width=True):
            if st.session_state.edit_history:
                op = st.session_state.edit_history.pop()
                apply_edit_operation(op, use_new_value=False)
                st.session_state.redo_history.append(op)
                if len(st.session_state.redo_history) > DEFAULT_EDIT_HISTORY_LIMIT:
                    st.session_state.redo_history.pop(0)
                st.rerun()

    with action_col2:
        if st.button("↷ Redo", use_container_width=True):
            if st.session_state.redo_history:
                op = st.session_state.redo_history.pop()
                apply_edit_operation(op, use_new_value=True)
                st.session_state.edit_history.append(op)
                if len(st.session_state.edit_history) > DEFAULT_EDIT_HISTORY_LIMIT:
                    st.session_state.edit_history.pop(0)
                st.rerun()

    with action_col3:
        if st.button("✅ 編集内容を結果に反映", type="primary"):
            with st.spinner("結果画像を更新中..."):
                st.session_state.result_pixel = processor.create_pixel_image(
                    st.session_state.label_image,
                    st.session_state.mapped_colors
                )
                st.session_state.color_counts = processor.create_color_counts(
                    st.session_state.label_image,
                    st.session_state.mapped_colors
                )
                st.session_state.edit_history.clear()
                st.session_state.redo_history.clear()

    reset_col1, reset_col2 = st.columns([0.2, 0.8])
    with reset_col1:
        if st.button("↩️ リセット"):
            st.session_state.label_image = st.session_state.original_label_image.copy()
            st.session_state.mapped_image = st.session_state.original_mapped_image.copy()
            st.session_state.result_pixel = processor.create_pixel_image(
                st.session_state.label_image,
                st.session_state.mapped_colors
            )
            st.session_state.color_counts = processor.create_color_counts(
                st.session_state.label_image,
                st.session_state.mapped_colors
            )
            st.session_state.edit_history.clear()
            st.session_state.redo_history.clear()
            st.rerun()


def main():
    """メインアプリケーション"""
    # ページ設定
    st.set_page_config(
        page_title="編み図メーカー",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 画像・キャンバス表示が親幅を超えた場合の表示崩れを抑える
    st.markdown(
        """
<style>
img, canvas {
  max-width: 100% !important;
  height: auto !important;
}

/* サイドバーの終了ボタンを最下部に固定 */
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

section[data-testid="stSidebar"] .st-key-exit_app_button {
    margin-top: auto;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # タイトル
    st.title("🎨 Image to Pixels Converter")
    st.markdown("画像を編み図で使えるドット絵に変換します")
    st.markdown("ユザワヤ(Yuzawaya) 毛糸 mansell をベースにした色変換を行います")

    st.sidebar.markdown("### 使い方")
    st.sidebar.caption("1) 画像を選択 -> 2) CSVを選択(任意) -> 3) 範囲を選択(任意) -> 4) 設定調整 -> 5) 処理実行")

    # サイドバー: Step 1 -> Step 4
    uploaded_file = upload_image_section()
    csv_bytes = upload_csv_section()
    params = setup_sidebar()

    st.sidebar.header("🚀 Step 5: 処理実行")
    run_clicked = st.sidebar.button(
        "処理を開始",
        use_container_width=True,
        type="primary",
        disabled=uploaded_file is None,
        key="run_process_button"
    )
    if uploaded_file is None:
        st.sidebar.caption("画像をアップロードすると実行できます。")

    render_exit_button()

    # メインコンテンツエリア
    if uploaded_file is not None:
        # 画像の読み込み
        image_data = uploaded_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        src_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 読み込み失敗のチェック
        if src_image is None:
            st.error("画像の読み込みに失敗しました。別のファイルを試してください。")
            return

        # セッション状態の初期化
        init_session_state(src_image)

        # 元画像の表示エリア
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            
            # 画像を表示用にリサイズ
            # 幅は実際のcol1の幅を最大値とする
            display_image, display_scale = resize_for_display(
                src_image.copy(),
                max_width=MAX_ROI_DISPLAY_WIDTH
            )
            
            # ROI選択UI
            render_roi_selection_ui(src_image.shape, display_image, display_scale)

        with col2:
            st.subheader("🧭 操作ガイド")
            roi_width, roi_height, roi_label = get_effective_roi_dimensions(src_image.shape)

            # 実処理側（ImageToPixels.create_label_image）では、ROI画像は
            # 最大辺が 2048px になるようにリサイズされてからセル数が決定される。
            # 表示用のセル数推定でも同じルールを適用することで、表示と実処理の差異をなくす。
            max_side = max(roi_width, roi_height)
            if max_side > 2048:
                scale = 2048 / max_side
                scaled_roi_width = int(round(roi_width * scale))
                scaled_roi_height = int(round(roi_height * scale))
            else:
                scaled_roi_width = roi_width
                scaled_roi_height = roi_height

            estimated_vertical_cells = estimate_vertical_cells(
                target_width=scaled_roi_width,
                target_height=scaled_roi_height,
                horizontal_cells=params["number_of_line_cells"],
                cell_width=params["cell_width"],
                cell_height=params["cell_height"],
            )

            st.info("Step 3: 必要なら左画像で範囲を選択し、サイドバーの『処理を開始』を押してください。")
            st.markdown(
                f"""
**現在の設定**
- 色数: {params['colors_number']}
- 対象: {roi_label} ({roi_width}×{roi_height}px)
- セル数: 横 {params['number_of_line_cells']} × 縦 {estimated_vertical_cells}
- ノイズ除去: {'ON' if params['denoise'] else 'OFF'}
"""
            )

        # 処理ボタン（サイドバーから）
        if run_clicked:
            with st.spinner("処理中..."):
                try:
                    # 処理対象の画像を決定
                    process_image = src_image.copy()
                    process_image = process_selected_roi(src_image, process_image)
                    
                    if process_image is None:
                        return
                    
                    # ImageToPixelsインスタンスの作成
                    processor = ImageToPixels()
                    
                    # パラメータの設定
                    processor.cell_height = params["cell_height"]
                    processor.cell_width = params["cell_width"]
                    processor.line_thickness = params["line_thickness"]
                    processor.thick_line_thickness = params["thick_line_thickness"]
                    processor.thick_line_interval = params["thick_line_interval"]
                    processor.colors_number = params["colors_number"]
                    processor.number_of_line_cells = params["number_of_line_cells"]
                    processor.denoise = params["denoise"]

                    st.session_state.processor = processor

                    # 処理実行（csv_bytes は bytes）
                    label_image, mapped_colors = processor.create_label_image(
                        process_image, csv_bytes)

                    st.session_state.label_image = label_image
                    st.session_state.original_label_image = label_image.copy()
                    st.session_state.mapped_colors = mapped_colors
                    st.session_state.mapped_image = processor.create_mapped_image(label_image, mapped_colors)
                    st.session_state.original_mapped_image = st.session_state.mapped_image.copy()
                    st.session_state.last_click = None
                    st.session_state.edit_history = []
                    st.session_state.redo_history = []

                    pixel = processor.create_pixel_image(label_image, mapped_colors)
                    color_counts = processor.create_color_counts(label_image, mapped_colors)

                    st.session_state.result_pixel = pixel
                    st.session_state.color_counts = color_counts
                    st.session_state.active_view = "結果"
                    st.success("処理完了！")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

        # 結果/編集の表示（再実行しても選択状態を保持）
        st.markdown("---")
        if "active_view" not in st.session_state:
            st.session_state.active_view = "結果"

        st.markdown("#### 表示モード")
        st.radio(
            "表示モード",
            ["結果", "編集"],
            key="active_view",
            horizontal=True,
            label_visibility="collapsed"
        )

        if st.session_state.active_view == "結果":
            if "result_pixel" in st.session_state:
                render_result_image()
                render_details_section(src_image)
            else:
                st.info("処理後に結果が表示されます。サイドバーの『処理を開始』を押してください。")
        else:
            if "label_image" in st.session_state and "mapped_colors" in st.session_state:
                render_edit_section()
            else:
                st.info("処理後に編集できます。まずは処理を実行してください。")

    else:
        # アップロード待機画面
        st.info("📁 サイドバーから画像ファイルをアップロードしてください")


if __name__ == "__main__":
    main()
