# -*- coding: utf-8 -*-
"""Plate Recognition with EasyOCR + GUI hiển thị ảnh và bảng thông tin"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageFont, ImageDraw, Image, ImageTk
import numpy as np
from easyocr import Reader
import cv2
from datetime import datetime

# ===============================
# Hàm chọn ảnh và nhận diện biển số
# ===============================
def open_and_process_image():
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.png;*.bmp")])
    if not file_path:
        return

    # Đọc ảnh
    img = cv2.imread(file_path)
    img = cv2.resize(img, (800, 600))

    # Font hiển thị chữ
    fontpath = "./arial.ttf"
    font = ImageFont.truetype(fontpath, 32)
    b, g, r, a = 0, 255, 0, 0

    # Tiền xử lý ảnh
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 200)

    # Tìm contour biển số
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    number_plate_shape = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approximation) == 4:
            number_plate_shape = approximation
            break

    # Nhận dạng EasyOCR
    if number_plate_shape is not None:
        (x, y, w, h) = cv2.boundingRect(number_plate_shape)
        number_plate = grayscale[y:y + h, x:x + w]

        reader = Reader(['en'])
        detection = reader.readtext(number_plate)

        if len(detection) == 0:
            plate_text = "Không thấy bảng số xe"
        else:
            detection_sorted = sorted(detection, key=lambda x: x[0][0][1])
            text_lines = [d[1] for d in detection_sorted]
            plate_text = " ".join(text_lines)
            cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
    else:
        plate_text = "Không tìm thấy vùng biển số!"

    # Vẽ chữ lên ảnh
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((150, 500), plate_text, font=font, fill=(b, g, r, a))

    # Hiển thị ảnh trên GUI
    img_tk = ImageTk.PhotoImage(img_pil)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Tạo dữ liệu bảng
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    for i in tree.get_children():
        tree.delete(i)  # xóa dữ liệu cũ
    tree.insert("", "end", values=(plate_text, time_str, date_str))


# ===============================
# GUI chính
# ===============================
root = tk.Tk()
root.title("Plate Recognition GUI")

# Nút chọn ảnh
btn_open = tk.Button(root, text="Chọn ảnh", command=open_and_process_image)
btn_open.pack(pady=10)

# Label hiển thị ảnh
label_img = tk.Label(root)
label_img.pack()

# Bảng hiển thị Biển số – Thời gian – Ngày
columns = ("Biển số", "Thời gian", "Ngày")
tree = ttk.Treeview(root, columns=columns, show="headings", height=1)
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=200)
tree.pack(pady=10)

root.mainloop()
