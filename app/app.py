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

        # セッション状態の初期化
        if "roi_p1" not in st.session_state:
            st.session_state.roi_p1 = None
            st.session_state.roi_p2 = None
            st.session_state.roi_selecting_point = None  # None, "p1", "p2" の3値
            st.session_state.last_click_coords = None  # 前回のクリック座標
            # デフォルト：画像全体
            height, width = src_image.shape[:2]
            st.session_state.roi_rect = (0, 0, width, height)

        # 元画像の表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            
            # 矩形選択UIセクション（折りたたみ可能）
            with st.expander("🔲 矩形領域選択（オプション）", expanded=False):
                st.caption("デフォルトでは画像全体を処理します。特定の領域のみを処理したい場合に設定してください。")
                
                # 選択状態の表示
                p1_status = "✅" if st.session_state.roi_p1 else "⭕"
                p2_status = "✅" if st.session_state.roi_p2 else "⭕"
                
                select_col1, select_col2, select_col3 = st.columns(3)
                
                with select_col1:
                    button_text = f"📍 左上 {p1_status}"
                    if st.button(button_text, use_container_width=True, type="secondary", key="btn_p1"):
                        st.session_state.roi_selecting_point = "p1"
                        st.session_state.last_click_coords = None  # 前回クリック座標をリセット
                        st.rerun()  # ボタンクリック時に画面更新して古い座標をクリア
                
                with select_col2:
                    button_text = f"📍 右下 {p2_status}"
                    if st.button(button_text, use_container_width=True, type="secondary", key="btn_p2"):
                        st.session_state.roi_selecting_point = "p2"
                        st.session_state.last_click_coords = None  # 前回クリック座標をリセット
                        st.rerun()  # ボタンクリック時に画面更新して古い座標をクリア
                
                with select_col3:
                    if st.button("🔄 リセット", use_container_width=True, key="btn_reset"):
                        st.session_state.roi_p1 = None
                        st.session_state.roi_p2 = None
                        st.session_state.last_click_coords = None
                        st.session_state.roi_selecting_point = None
                        height, width = src_image.shape[:2]
                        st.session_state.roi_rect = (0, 0, width, height)
                
                st.markdown(f"**選択状態**: 左上 {p1_status} `{st.session_state.roi_p1 if st.session_state.roi_p1 else '未選択'}` | 右下 {p2_status} `{st.session_state.roi_p2 if st.session_state.roi_p2 else '未選択'}`")
            
            # 画像表示とインタラクション
            display_image = src_image.copy()
            
            # 選択済みポイントを描画
            if st.session_state.roi_p1:
                cv2.circle(display_image, st.session_state.roi_p1, 8, (0, 255, 0), -1)
                cv2.putText(display_image, "P1(LT)", (st.session_state.roi_p1[0]+10, st.session_state.roi_p1[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if st.session_state.roi_p2:
                cv2.circle(display_image, st.session_state.roi_p2, 8, (255, 0, 0), -1)
                cv2.putText(display_image, "P2(RB)", (st.session_state.roi_p2[0]+10, st.session_state.roi_p2[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 両点が選択されたら矩形を描画
            if st.session_state.roi_p1 and st.session_state.roi_p2:
                p1 = st.session_state.roi_p1
                p2 = st.session_state.roi_p2
                x1, x2 = sorted([p1[0], p2[0]])
                y1, y2 = sorted([p1[1], p2[1]])
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # 矩形内を半透明に
                overlay = display_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                display_image = cv2.addWeighted(overlay, 0.15, display_image, 0.85, 0)
            
            # 画像を表示用にリサイズ（最大幅800px）
            orig_height, orig_width = display_image.shape[:2]
            max_display_width = 800
            if orig_width > max_display_width:
                display_scale = max_display_width / orig_width
                display_width = max_display_width
                display_height = int(orig_height * display_scale)
                display_resized = cv2.resize(display_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            else:
                display_scale = 1.0
                display_resized = display_image
            
            # 画像をクリック可能にして座標取得
            coords = streamlit_image_coordinates(cv2.cvtColor(display_resized, cv2.COLOR_BGR2RGB), key="roi_selector")
            
            if coords is not None and "x" in coords and "y" in coords:
                # クリック座標を元の画像サイズに変換
                click_point = (int(coords["x"] / display_scale), int(coords["y"] / display_scale))
                
                # 前回とは異なるクリックかどうかを確認
                if click_point != st.session_state.last_click_coords:
                    st.session_state.last_click_coords = click_point
                    
                    # roi_selecting_point が設定されている場合のみ座標を保存
                    if st.session_state.roi_selecting_point == "p1":
                        st.session_state.roi_p1 = click_point
                        st.session_state.roi_selecting_point = None  # 入力モード解除
                        st.success(f"✅ 左上: {click_point}")
                    elif st.session_state.roi_selecting_point == "p2":
                        st.session_state.roi_p2 = click_point
                        st.session_state.roi_selecting_point = None  # 入力モード解除
                        st.success(f"✅ 右下: {click_point}")
                    
                    # 両点が選択されたら矩形を確定（p1, p2どちらを選択した場合でもチェック）
                    if st.session_state.roi_p1 and st.session_state.roi_p2:
                        p1 = st.session_state.roi_p1
                        p2 = st.session_state.roi_p2
                        x1, x2 = sorted([p1[0], p2[0]])
                        y1, y2 = sorted([p1[1], p2[1]])
                        st.session_state.roi_rect = (x1, y1, x2, y2)
                    
                    st.rerun()

        # 処理ボタン
        if st.button("🚀 処理実行", use_container_width=True, type="primary"):
            with st.spinner("処理中..."):
                try:
                    # 処理対象の画像を決定
                    process_image = src_image.copy()
                    
                    # 矩形が選択されている場合、その領域のみを抽出
                    if st.session_state.roi_rect:
                        x1, y1, x2, y2 = st.session_state.roi_rect
                        
                        # 画像全体かどうかをチェック
                        is_full_image = (x1 == 0 and y1 == 0 and 
                                       x2 == src_image.shape[1] and y2 == src_image.shape[0])
                        
                        if not is_full_image:
                            # 矩形サイズの検証
                            if x1 >= x2 or y1 >= y2:
                                st.error(f"⚠️ 矩形のサイズが不正です: ({x1}, {y1}) - ({x2}, {y2})")
                            elif (x2 - x1) < 2 or (y2 - y1) < 2:
                                st.error(f"⚠️ 矩形が小さすぎます: 幅{x2-x1}px, 高さ{y2-y1}px（最小2px必要）")
                            else:
                                process_image = src_image[y1:y2, x1:x2].copy()
                                st.session_state.roi_offset = (x1, y1)  # オフセットを保存
                                
                                if process_image.size == 0:
                                    st.error("⚠️ 抽出した画像が空です")
                                else:
                                    st.info(f"📍 処理対象: 選択領域 位置({x1}, {y1}) サイズ {x2-x1}×{y2-y1}")
                        else:
                            st.session_state.roi_offset = (0, 0)
                            st.info("📍 処理対象: 画像全体")
                    else:
                        st.session_state.roi_offset = (0, 0)
                    
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

                    st.session_state.processor = processor

                    # 処理実行
                    label_image, mapped_colors = processor.create_label_image(process_image)

                    st.session_state.label_image = label_image
                    st.session_state.original_label_image = label_image.copy()
                    st.session_state.mapped_colors = mapped_colors
                    st.session_state.last_click = None

                    pixel = processor.create_pixel_image(label_image, mapped_colors)
                    color_counts = processor.create_color_counts(label_image, mapped_colors)

                    st.session_state.result_pixel = pixel
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
            
            # 矩形選択情報の表示（選択されている場合のみ）
            if "roi_rect" in st.session_state and st.session_state.roi_rect:
                x1, y1, x2, y2 = st.session_state.roi_rect
                roi_width = x2 - x1
                roi_height = y2 - y1
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

            # 処理結果のダウンロード
            st.markdown("---")
            st.subheader("📥 結果をダウンロード")
            
            # mapped_colors を CSV データに変換
            colors_data = []
            for idx, color in enumerate(st.session_state.mapped_colors):
                colors_data.append({
                    "色番": idx,
                    "色名": color.type,
                    "色コード": color.color_number,
                    "R": color.rgb[0],
                    "G": color.rgb[1],
                    "B": color.rgb[2]
                })
            colors_df = pd.DataFrame(colors_data)
            
            # ダウンロード用CSV
            colors_csv = colors_df.to_csv(index=False, encoding='utf-8-sig')
            
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
