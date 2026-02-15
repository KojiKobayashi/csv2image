import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# srcフォルダをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image2cells import ImageToPixels

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None


def _ensure_tmp_dir():
    Path("./tmp").mkdir(parents=True, exist_ok=True)


def main():
    # ページ設定
    st.set_page_config(
        page_title="CSV to Image - Image to Pixels",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # タイトル
    st.title("🎨 Image to Pixels Converter")
    st.markdown("画像をドット絵に変換します")

    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # パラメータの調整
    colors_number = st.sidebar.slider("量子化する色数", min_value=4, max_value=64, value=12, step=1)
    number_of_line_cells = st.sidebar.slider("横セル数", min_value=16, max_value=256, value=64, step=8)
    cell_height = st.sidebar.slider("セル高さ", min_value=10, max_value=100, value=27, step=1)
    cell_width = st.sidebar.slider("セル幅", min_value=10, max_value=100, value=33, step=1)
    line_thickness = st.sidebar.slider("通常グリッド線の太さ", min_value=1, max_value=10, value=1, step=1)
    thick_line_thickness = st.sidebar.slider("太いグリッド線の太さ", min_value=2, max_value=10, value=3, step=1)
    thick_line_interval = st.sidebar.slider("太いグリッド線の間隔（セル数）", min_value=1, max_value=20, value=5, step=1)
    denoise = st.sidebar.checkbox("ノイズ除去を有効にする", value=False)

    # 画像アップロード
    st.sidebar.markdown("---")
    st.sidebar.header("📁 ファイル")
    uploaded_file = st.sidebar.file_uploader(
        "画像ファイルを選択",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    )

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

        # 元画像の表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            st.image(src_image, use_container_width=True, channels="BGR")

        # 処理ボタン
        if st.button("🚀 処理実行", use_container_width=True, type="primary"):
            with st.spinner("処理中..."):
                try:
                    # ImageToPixelsインスタンスの作成
                    processor = ImageToPixels()
                    
                    # パラメータの設定
                    processor.cell_height = cell_height
                    processor.cell_width = cell_width
                    processor.line_thickness = line_thickness
                    processor.thick_line_thickness = thick_line_thickness
                    processor.thick_line_interval = thick_line_interval
                    processor.colors_number = colors_number
                    processor.number_of_line_cells = number_of_line_cells
                    processor.denoise = denoise

                    # 一時的にファイルを保存して処理
                    # TODO オンメモリで処理できるようにする
                    _ensure_tmp_dir()
                    temp_image_path = "./tmp/temp_image.jpg"
                    cv2.imwrite(temp_image_path, src_image)

                    # 処理実行
                    # pixel, color_counts = processor.run(temp_image_path)

                    label_image, mapped_colors = processor.create_label_image(temp_image_path)
                    st.session_state.label_image = label_image
                    st.session_state.original_label_image = label_image.copy()
                    st.session_state.mapped_colors = mapped_colors
                    st.session_state.processor = processor
                    st.session_state.last_click = None

                    pixel = processor.create_pixel_image(label_image, mapped_colors)
                    color_counts = processor.create_color_counts(label_image, mapped_colors)

                    # セッションステートに結果を保存
                    st.session_state.result_pixel = pixel
                    # st.session_state.centers = centers
                    st.session_state.color_counts = color_counts
                    st.success("処理完了！")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

        # 結果の表示
        if "result_pixel" in st.session_state:
            with col2:
                st.subheader("処理結果")
                st.image(st.session_state.result_pixel, use_container_width=True, channels="BGR")

            # 詳細情報の表示
            st.markdown("---")
            st.subheader("📊 詳細情報")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("取得した色数", len(st.session_state.color_counts))
            
            with info_col3:
                st.metric("ピクセル総数", sum(c.count for c in st.session_state.color_counts))

            # 色カウント情報
            st.markdown("#### 🎨 色ごとのピクセル数")
            
            # 各色の情報を表示
            for idx, color in enumerate(st.session_state.color_counts):
                col1, col2, col3 = st.columns([0.1, 0.45, 0.45])
                
                with col1:
                    # BGR形式のRGBをRGB形式に変換して色見本を表示
                    rgb_color = f"rgb({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})"
                    st.markdown(
                        f'<div style="width: 40px; height: 40px; background-color: {rgb_color}; border: 1px solid #ccc; border-radius: 4px;"></div>',
                        unsafe_allow_html=True
                    )

                with col2:
                    st.text(f"**{color.type}** ({color.color_number})")
                
                with col3:
                    st.metric("ピクセル数", f"{color.count:,}", label_visibility="collapsed")

            # 処理結果の保存
            if st.button("💾 結果を保存", use_container_width=True):
                output_path = "output_pixelized.png"
                cv2.imwrite(output_path, st.session_state.result_pixel)
                st.success(f"{output_path}に保存しました")

            # 編集UI
            if "label_image" in st.session_state and "mapped_colors" in st.session_state:
                st.markdown("---")
                st.subheader("🖌️ 編集")

                if streamlit_image_coordinates is None:
                    st.warning("編集UIを使うには streamlit-image-coordinates の導入が必要です。")
                    return

                edit_scale = st.sidebar.slider("編集表示倍率", min_value=4, max_value=20, value=10, step=1)

                if "selected_color_idx" not in st.session_state:
                    st.session_state.selected_color_idx = 0

                st.markdown("#### 🎯 色の選択")
                palette_cols = st.columns(6)
                for idx, color in enumerate(st.session_state.mapped_colors):
                    with palette_cols[idx % 6]:
                        rgb_color = f"rgb({color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]})"
                        st.markdown(
                            f'<div style="width: 36px; height: 36px; background-color: {rgb_color}; border: 1px solid #ccc; border-radius: 4px;"></div>',
                            unsafe_allow_html=True
                        )
                        label = "選択中" if idx == st.session_state.selected_color_idx else "選択"
                        if st.button(label, key=f"palette_{idx}"):
                            st.session_state.selected_color_idx = idx

                selected_color = st.session_state.mapped_colors[st.session_state.selected_color_idx]
                st.markdown(
                    f"選択中: {selected_color.type} ({selected_color.color_number})"
                )

                label_image = st.session_state.label_image
                processor = st.session_state.get("processor", ImageToPixels())
                mapped_image = processor.create_mapped_image(label_image, st.session_state.mapped_colors)

                height, width = mapped_image.shape[:2]
                preview = cv2.resize(
                    mapped_image,
                    (width * edit_scale, height * edit_scale),
                    interpolation=cv2.INTER_NEAREST
                )
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

                st.markdown("#### 🧭 クリックで塗る")
                coords = streamlit_image_coordinates(preview_rgb, key="editor_canvas")
                if coords is not None and "x" in coords and "y" in coords:
                    click = (coords["x"], coords["y"])
                    if st.session_state.last_click != click:
                        st.session_state.last_click = click
                        cell_x = int(coords["x"] // edit_scale)
                        cell_y = int(coords["y"] // edit_scale)
                        if 0 <= cell_x < width and 0 <= cell_y < height:
                            st.session_state.label_image[cell_y, cell_x] = st.session_state.selected_color_idx
                            st.session_state.result_pixel = processor.create_pixel_image(
                                st.session_state.label_image,
                                st.session_state.mapped_colors
                            )
                            st.session_state.color_counts = processor.create_color_counts(
                                st.session_state.label_image,
                                st.session_state.mapped_colors
                            )
                            st.rerun()

                reset_col1, reset_col2 = st.columns([0.2, 0.8])
                with reset_col1:
                    if st.button("↩️ リセット"):
                        st.session_state.label_image = st.session_state.original_label_image.copy()
                        st.session_state.result_pixel = processor.create_pixel_image(
                            st.session_state.label_image,
                            st.session_state.mapped_colors
                        )
                        st.session_state.color_counts = processor.create_color_counts(
                            st.session_state.label_image,
                            st.session_state.mapped_colors
                        )

    else:
        # アップロード待機画面
        st.info("📁 サイドバーから画像ファイルをアップロードしてください")


if __name__ == "__main__":
    main()
