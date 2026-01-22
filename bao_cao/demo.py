# -*- coding: utf-8 -*-
"""Plate Recognition with EasyOCR + Send Signal to ESP32"""

from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import matplotlib.pyplot as plt
import datetime
import serial
import time

# ===============================
# Load file tỉnh -> dictionary
# ===============================
province_map = {}
with open("provinces.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            code, name = parts
            province_map[code] = name

# ===============================
# 1. Đọc ảnh và tiền xử lý
# ===============================
img = cv2.imread('20.jpg')
img = cv2.resize(img, (800, 600))

fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)

# ===============================
# 2. Tìm contour biển số
# ===============================
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

number_plate_shape = None

for c in contours:
    perimeter = cv2.arcLength(c, True)
    canh = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(canh) == 4:
        number_plate_shape = canh
        break

# ===============================
# 3. Nhận dạng
# ===============================
if number_plate_shape is not None:
    (x, y, w, h) = cv2.boundingRect(number_plate_shape)
    number_plate = grayscale[y:y + h, x:x + w]

    reader = Reader(['en'])
    detection = reader.readtext(number_plate)

    if len(detection) == 0:
        bien_so = "Không đọc được kí tự biển số"
        tinh = "Không rõ"

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((150, 500), bien_so, font=font, fill=(b, g, r, a))
        img_result = np.array(img_pil)
    else:
        cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
        detection_sorted = sorted(detection, key=lambda x: x[0][0][1])
        text_lines = [d[1] for d in detection_sorted]
        bien_so = " ".join(text_lines)

        # Lấy 2 ký tự đầu
        ma_tinh = bien_so[:2]
        tinh = province_map.get(ma_tinh, "Không rõ")

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((150, 500), bien_so, font=font, fill=(b, g, r, a))
        img_result = np.array(img_pil)

else:
    bien_so = "Không nhận dạng được vùng biển số"
    tinh = "Không rõ"

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((150, 500), bien_so, font=font, fill=(255, 0, 0, 0))
    img_result = np.array(img_pil)

import unicodedata

def remove_accents(input_str):
    # Loại bỏ dấu tiếng Việt
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

# ===============================
# Gửi tín hiệu sang ESP32
# ===============================
try:
    esp32 = serial.Serial('COM3', 115200, timeout=1)
    time.sleep(2)

    # Gửi biển số
    esp32.write(f"PLATE:{bien_so}\n".encode())

    # Gửi tỉnh: không dấu, chữ thường
    tinh_khong_dau = remove_accents(tinh).lower()
    esp32.write(f"PROVINCE:{tinh_khong_dau}\n".encode())

    # Gửi tín hiệu LED
    if "Không" in bien_so:
        esp32.write(b"LED2_ON\n")
    else:
        esp32.write(b"LED1_ON\n")

    esp32.close()

except Exception as e:
    print("Không thể kết nối ESP32:", e)

# ===============================
# 5. Hiển thị giao diện
# ===============================
now = datetime.datetime.now()
ngay = now.strftime("%d/%m/%Y")
gio = now.strftime("%H:%M:%S")

fig = plt.figure(figsize=(12, 6))

# Ảnh bên trái
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img_result)
ax1.axis('off')
ax1.set_title("Plate Detection Result")

# Thông tin bên phải
ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')

table_data = [
    ["Biển số", bien_so],
    ["Tỉnh", tinh],
    ["Ngày", ngay],
    ["Thời gian", gio],
]

table = ax2.table(
    cellText=table_data,
    colLabels=["Thông tin", "Giá trị"],
    loc='center',
    cellLoc='center'
)

table.scale(1, 1)
plt.tight_layout()
plt.show()
