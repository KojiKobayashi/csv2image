import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# srcフォルダをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image2cells import ImageToPixels


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
            st.image(src_image, use_column_width=True, channels="BGR")

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
                    temp_image_path = "./tmp/temp_image.jpg"
                    cv2.imwrite(temp_image_path, src_image)

                    # 処理実行
                    pixel, centers, color_counts = processor.run(temp_image_path)

                    # セッションステートに結果を保存
                    st.session_state.result_pixel = pixel
                    st.session_state.centers = centers
                    st.session_state.color_counts = color_counts

                    st.success("処理完了！")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

        # 結果の表示
        if "result_pixel" in st.session_state:
            with col2:
                st.subheader("処理結果")
                st.image(st.session_state.result_pixel, use_column_width=True, channels="BGR")

            # 詳細情報の表示
            st.markdown("---")
            st.subheader("📊 詳細情報")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("取得した色数", len(st.session_state.centers))
            
            with info_col3:
                st.metric("ピクセル総数", sum(c.count for c in st.session_state.color_counts))

            # 色カウント情報
            st.markdown("#### 🎨 色ごとのピクセル数")
            
            # 各色の情報を表示
            for idx, color in enumerate(st.session_state.color_counts):
                col1, col2, col3 = st.columns([0.1, 0.45, 0.45])
                
                with col1:
                    # BGR形式のRGBをRGB形式に変換して色見本を表示
                    rgb_color = f"rgb({color.rgb[2]}, {color.rgb[1]}, {color.rgb[0]})"
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

    else:
        # アップロード待機画面
        st.info("📁 サイドバーから画像ファイルをアップロードしてください")


if __name__ == "__main__":
    main()
