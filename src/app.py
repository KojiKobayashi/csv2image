'''
app の Docstring
以下の機能を持つGUIアプリケーション:
- ユーザーに画像を選択させる
- 選択された画像を表示する
- 矩形をユーザーに一つ囲わせる
- 囲まれた矩形の座標を取得し、表示する
- 画像処理用のパラメタをユーザーに入力させる
    - 設定師はデフォルトをsettings.pyから取得する
- 画像処理を実行し、結果を表示する
- パラメタが更新させるたび、画像処理を再実行し、結果を更新する

TODO
- カラーリスト表示
- パラメタ反映
- 色数表示
- ピクセルごとの色入れ替え
'''

import tkinter as tk
from tkinter import filedialog, simpledialog, Canvas, Scrollbar, messagebox, ttk
import cv2
import sys
from image2cells import display_image, run_image
import settings
import threading
import numpy as np

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

         # 縦スクロールバー
        vscrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL)
        vscrollbar.pack(fill=tk.Y, side=tk.RIGHT, expand=False)
        # 横スクロールバー
        hscrollbar = ttk.Scrollbar(root, orient=tk.HORIZONTAL)
        hscrollbar.pack(fill=tk.X, side=tk.BOTTOM, expand=False)

        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack(expand=True, fill=tk.BOTH)

                # スクロールバーを Canvas に関連付け
        vscrollbar.config(command=self.canvas.yview)
        hscrollbar.config(command=self.canvas.xview)


        # self.scrollbar = Scrollbar(root, command=self.canvas.yview)
        # self.scrollbar.pack(side='right', fill='y')

        # self.canvas.config(yscrollcommand=self.scrollbar.set)

        self.image = None
        self.roi = None
        self.roi_coords = None

        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack()


        self.param_button = tk.Button(root, text="Run", command=self.run_image)
        self.param_button.pack()
                
        self.set_roi_button = tk.Button(root, text="Set Rctangle", command=self.set_roi)
        self.set_roi_button.pack()


        # parameters
        self.denoise_checkbox = tk.Checkbutton(root, text="Denoise?", command=self.set_denoise)
        self.denoise_checkbox.pack()

        # 最小値2, 最大値30のスライダーを追加
        self.cell_height_slider = tk.Scale(root, from_=2, to=30, orient=tk.HORIZONTAL, label="number of colors", command=self.set_number_of_colors)
        self.cell_height_slider.pack()

        self.cell_height_slider = tk.Scale(root, from_=10, to=1000, orient=tk.HORIZONTAL, label="number_of_line_cells", command=self.set_number_of_line_cells)
        self.cell_height_slider.pack()

        self.number_of_line_cells = settings.number_of_line_cells
        self.number_of_colors = settings.number_of_colors
        self.denoise = settings.denoise

        # 空の画像を用意
        self.disp = None
        self.image = None
        self.roi = None
        self.color_list = None

    def run_image(self):
        if self.roi is None:
            return
        proc_image, color_list = run_image(self.roi)
        self.disp = proc_image
        self.color_list = color_list

        if proc_image is None:
            return

        byte = cv2.imencode('.png', proc_image)[1].tobytes()
        self.photo_image = tk.PhotoImage(data=byte)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        # canvasにroiを表示
        # photoImageに変換
        self.show_image(self.disp)


    def _resize_image(self, image):
        # canvas に入るよう画像をリサイズ
        resize_rate = (float)(self.canvas.winfo_width()/image.shape[1])
        return cv2.resize(image, None, fx=resize_rate, fy=resize_rate, interpolation=cv2.INTER_LINEAR)

    def set_roi(self):
        if self.image is None:
            return
        
        r = cv2.selectROI("Select ROI", self.image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        if r == (0,0,0,0):
            r = (0, 0, self.image.shape[1], self.image.shape[0])
        
        self.roi_coords = r
        self.roi = self.image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        self.disp = self.roi
        # display_image(self.roi)

        # canvasにroiを表示
        # photoImageに変換
        self.show_image(self.roi)
        # roi_byte = cv2.imencode('.png', self.roi)[1].tobytes()
        # self.photo_image = tk.PhotoImage(data=roi_byte)
        # self.canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        # self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def set_number_of_line_cells(self, val):
        self.number_of_line_cells = int(val)

    def set_number_of_colors(self, val):
        self.number_of_colors = int(val)
        # messagebox.showinfo("Info", f"Cell height set to {settings.cell_height}")

    def set_denoise(self):
        self.denoise = not settings.denoise
        # messagebox.showinfo("Info", f"Denoise set to {settings.denoise}")
    

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            return
        
        self.image = cv2.imread(file_path)
        if self.image is None:
            messagebox.showerror("Error", "Image not found or unable to load.")
            return
        self.roi = self.image
        self.disp = self.image

        self.show_image(self.image)
        
        # resize = self._resize_image(self.image)
        # byte = cv2.imencode('.png', resize)[1].tobytes()
        # self.photo_image = tk.PhotoImage(data=byte)
        # self.canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        # self.canvas.config(scrollregion=self.canvas.bbox("all"))


    def show_image(self, img):
        resize = self._resize_image(img)
        byte = cv2.imencode('.png', resize)[1].tobytes()
        self.photo_image = tk.PhotoImage(data=byte)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    # def process_image(self):
    #     if self.roi is None:
    #         messagebox.showerror("Error", "No ROI selected.")
    #         return
        
    #     threading.Thread(target=self._process_image_thread).start()
    
    # def _process_image_thread(self):
    #     processed_image = run(self.roi)
    #     display_image(processed_image)
    #     self.canvas.create_image(0, 0, anchor='nw', image=processed_image)
    #     self.canvas.config(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
    