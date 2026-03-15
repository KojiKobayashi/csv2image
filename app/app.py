import os
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import io
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
MAX_EDIT_DISPLAY_WIDTH = 700
CIRCLE_RADIUS = 8
CIRCLE_THICKNESS = -1
RECTANGLE_THICKNESS = 3
OVERLAY_ALPHA = 0.15

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
    return colors_df.to_csv(index=False, encoding='utf-8-sig')


def get_rect_dimensions(rect:tuple[int, int, int, int]) -> tuple[int, int]:
    """矩形情報から幅と高さを取得"""
    x1, y1, x2, y2 = rect
    return x2 - x1, y2 - y1


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
    st.sidebar.header("⚙️ Step 3: 変換設定")
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
    
    # ダウンロード用画像
    _, img_bytes = cv2.imencode('.png', st.session_state.result_pixel)
    img_buffer = io.BytesIO(img_bytes)
    
    # ダウンロードボタン
    col_img, col_csv = st.columns(2)
    
    with col_img:
        st.download_button(
            label="🖼️ ドット絵をダウンロード",
            data=img_buffer,
            file_name="result_pixelized.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col_csv:
        st.download_button(
            label="📊 色情報をダウンロード",
            data=colors_csv,
            file_name="color_palette.csv",
            mime="text/csv",
            use_container_width=True
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
                        st.session_state.label_image[cell_y, cell_x] = selected_idx
                        selected_bgr = tuple(reversed(st.session_state.mapped_colors[selected_idx].rgb))
                        st.session_state.mapped_image[cell_y, cell_x] = selected_bgr
                        st.session_state.edit_history.append({
                            "x": cell_x,
                            "y": cell_y,
                            "prev": prev_idx,
                            "new": selected_idx
                        })
                        if len(st.session_state.edit_history) > DEFAULT_EDIT_HISTORY_LIMIT:
                            st.session_state.edit_history.pop(0)
                        st.session_state.redo_history.clear()
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

    has_pending_edits = len(st.session_state.edit_history) > 0
    if has_pending_edits:
        st.warning("未反映の編集があります。必要なら「✅ 編集内容を結果に反映」を押してください。")
        st.caption(
            f"未反映の編集数: {len(st.session_state.edit_history)} / 履歴上限: {DEFAULT_EDIT_HISTORY_LIMIT}"
        )
    
    action_col1, action_col2, action_col3 = st.columns([0.2, 0.2, 0.3])
    with action_col1:
        if st.button("↶ Undo", use_container_width=True):
            if st.session_state.edit_history:
                op = st.session_state.edit_history.pop()
                x, y, prev_idx = op["x"], op["y"], op["prev"]
                st.session_state.label_image[y, x] = prev_idx
                prev_bgr = tuple(reversed(st.session_state.mapped_colors[prev_idx].rgb))
                st.session_state.mapped_image[y, x] = prev_bgr
                st.session_state.redo_history.append(op)
                if len(st.session_state.redo_history) > DEFAULT_EDIT_HISTORY_LIMIT:
                    st.session_state.redo_history.pop(0)
                st.rerun()

    with action_col2:
        if st.button("↷ Redo", use_container_width=True):
            if st.session_state.redo_history:
                op = st.session_state.redo_history.pop()
                x, y, new_idx = op["x"], op["y"], op["new"]
                st.session_state.label_image[y, x] = new_idx
                new_bgr = tuple(reversed(st.session_state.mapped_colors[new_idx].rgb))
                st.session_state.mapped_image[y, x] = new_bgr
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

/* 結果/編集タブを見やすくする */
button[data-baseweb="tab"] p {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"] {
    min-height: 44px !important;
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # タイトル
    st.title("🎨 Image to Pixels Converter")
    st.markdown("画像をドット絵に変換します")

    st.sidebar.markdown("### 使い方")
    st.sidebar.caption("1) 画像を選択 -> 2) 範囲を選択(任意) -> 3) 設定調整 -> 4) 処理実行")

    # サイドバー: Step 1 -> Step 3
    uploaded_file = upload_image_section()
    params = setup_sidebar()

    st.sidebar.header("🚀 Step 4: 処理実行")
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
            st.info("Step 2: 必要なら左画像で範囲を選択し、サイドバーの『処理を開始』を押してください。")
            st.markdown(
                f"""
**現在の設定**
- 色数: {params['colors_number']}
- 横セル数: {params['number_of_line_cells']}
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

                    # 処理実行
                    label_image, mapped_colors = processor.create_label_image(process_image, MERINO_RAINBOW_CSV)

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
                    st.success("処理完了！")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

        # 結果の表示
        st.markdown("---")
        tab_result, tab_edit = st.tabs(["結果", "編集"])

        with tab_result:
            if "result_pixel" in st.session_state:
                render_result_image()
                render_details_section(src_image)
            else:
                st.info("処理後に結果が表示されます。サイドバーの『処理を開始』を押してください。")

        with tab_edit:
            if "label_image" in st.session_state and "mapped_colors" in st.session_state:
                render_edit_section()
            else:
                st.info("処理後に編集できます。まずは処理を実行してください。")

    else:
        # アップロード待機画面
        st.info("📁 サイドバーから画像ファイルをアップロードしてください")


if __name__ == "__main__":
    main()
