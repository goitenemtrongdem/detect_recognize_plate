# -*- coding: utf-8 -*-
"""Plate Recognition with EasyOCR + Send Signal to ESP32
   - Xử lý: loại bỏ [], Z->2, I->1, ký tự đặc biệt -> '-'
   - Xác định tỉnh SAU khi đã chuẩn hoá biển số
   - Gửi sang ESP32: chỉ bỏ dấu (không chuyển hoa/thường)
"""

from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2
import matplotlib.pyplot as plt
import datetime
import serial
import time
import unicodedata
import os

# ===============================
# Hàm bỏ dấu (không đổi hoa/thường)
# ===============================
def remove_accents(input_str: str) -> str:
    nfkd = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

# ===============================
# Load file tỉnh -> dictionary (hỗ trợ "10:Ha Noi" hoặc "10 Ha Noi")
# ===============================
province_map = {}
provinces_file = "provinces.txt"
if os.path.exists(provinces_file):
    with open(provinces_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # hỗ trợ cả "10:Ha Noi" hoặc "10 Ha Noi"
            if ":" in line:
                code, name = line.split(":", 1)
            else:
                parts = line.split(None, 1)  # split by any whitespace, max 1
                if len(parts) < 2:
                    continue
                code, name = parts
            province_map[code.strip()] = name.strip()
else:
    print(f"Warning: {provinces_file} not found. province_map empty.")

# ===============================
# 1. Đọc ảnh và tiền xử lý
# ===============================
img_path = "15.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Ảnh không tìm thấy: {img_path}")

img = cv2.resize(img, (800, 600))

fontpath = "./arial.ttf"
if not os.path.exists(fontpath):
    # dùng font mặc định PIL nếu không có arial.ttf
    font = ImageFont.load_default()
else:
    font = ImageFont.truetype(fontpath, 32)

color_text = (0, 255, 0)  # RGB

grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)

# ===============================
# 2. Tìm contour biển số
# ===============================
contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

number_plate_shape = None

for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        number_plate_shape = approx
        break

# ===============================
# 3. Nhận dạng với EasyOCR
# ===============================
reader = Reader(['en'], gpu=False)

if number_plate_shape is not None:
    (x, y, w, h) = cv2.boundingRect(number_plate_shape)
    # cẩn thận tránh vùng ngoài ảnh
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    number_plate = grayscale[y1:y2, x1:x2]

    detection = reader.readtext(number_plate)

    if len(detection) == 0:
        bien_so = "Không đọc được kí tự biển số"
        tinh = "Không rõ"
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((150, 500), bien_so, font=font, fill=color_text)
        img_result = np.array(img_pil)
    else:
        cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
        detection_sorted = sorted(detection, key=lambda x: x[0][0][1])
        text_lines = [d[1] for d in detection_sorted]
        original_text = " ".join(text_lines).strip()

        # ===============================
        # XỬ LÝ BIỂN SỐ: Bỏ [], Z->2, I->1, ký tự đặc biệt -> '-'
        # ===============================
        bien_so = original_text

        # (a) bỏ ngoặc vuông
        bien_so = bien_so.replace("[", "").replace("]", "")

        # (b) Z/z -> 2
        bien_so = bien_so.replace("Z", "2").replace("z", "2")

        # (c) I/i -> 1
        bien_so = bien_so.replace("I", "1").replace("i", "1")
        bien_so = bien_so.replace("g", "9").replace("G", "G")  # thêm thay g/G bằng 9
        bien_so = bien_so.replace("J", "3").replace("j", "3") 
        bien_so = bien_so.replace("p", "B").replace("P", "B").replace("a", "A")  # p/b/P -> B
        # (d) các ký tự đặc biệt -> '-'
        special_chars = [":", "*", "/", "\\", "|",  ",", ";"]
        for ch in special_chars:
            bien_so = bien_so.replace(ch, "-")

        # (e) gộp nhiều dấu '-' liên tiếp thành 1 (tùy chọn nhưng hữu ích)
        while "--" in bien_so:
            bien_so = bien_so.replace("--", "-")

        # Bản hiển thị (giữ nguyên hoa/thường)
        bien_so_hienthi = bien_so

        # ===============================
        # XÁC ĐỊNH TỈNH SAU KHI ĐÃ XỬ LÝ
        # ===============================
        ma_tinh = None
        tinh = "Không rõ"

        # 1) thử lấy 2 ký tự đầu nếu là số
        if len(bien_so_hienthi) >= 2 and bien_so_hienthi[0:2].isdigit():
            candidate = bien_so_hienthi[0:2]
            if candidate in province_map:
                ma_tinh = candidate
                tinh = province_map[ma_tinh]
        # 2) otherwise tìm bất kỳ vị trí nào xuất hiện 2 chữ số liên tiếp
        if ma_tinh is None:
            for i in range(len(bien_so_hienthi) - 1):
                block = bien_so_hienthi[i:i+2]
                if block.isdigit() and block in province_map:
                    ma_tinh = block
                    tinh = province_map[ma_tinh]
                    break

        # Vẽ và hiển thị
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((150, 500), bien_so_hienthi, font=font, fill=color_text)
        img_result = np.array(img_pil)

else:
    bien_so = "Không nhận dạng được vùng biển số"
    tinh = "Không rõ"
    bien_so_hienthi = bien_so

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((150, 500), bien_so_hienthi, font=font, fill=(255, 0, 0))
    img_result = np.array(img_pil)

import serial
import time

try:
    # --- Arduino UNO dùng 9600 ---
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)  # đợi Arduino reset

    # Gửi biển số (đã bỏ dấu)
    plate_to_send = remove_accents(bien_so_hienthi)
    arduino.write(f"PLATE:{plate_to_send}\n".encode())

    # Gửi tỉnh (đã bỏ dấu)
    province_to_send = remove_accents(tinh)
    arduino.write(f"PROVINCE:{province_to_send}\n".encode())

    # Kiểm tra lỗi: nếu có chữ "khong" thì LED2_ON
    plate_check = remove_accents(bien_so_hienthi).lower()
    province_check = remove_accents(tinh).lower()

    if "khong" in plate_check or "khong" in province_check:
        arduino.write(b"LED2_ON\n")   # không nhận diện
    else:
        arduino.write(b"LED1_ON\n")   # nhận diện OK

    arduino.close()

except Exception as e:
    print("Không thể kết nối Arduino:", e)


# ===============================
# 5. Hiển thị giao diện kết quả
# ===============================
now = datetime.datetime.now()
ngay = now.strftime("%d/%m/%Y")
gio = now.strftime("%H:%M:%S")

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img_result)
ax1.axis('off')
ax1.set_title("Plate Detection Result")

ax2 = fig.add_subplot(1, 2, 2)
ax2.axis('off')

table_data = [
    ["Biển số", bien_so_hienthi],
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
