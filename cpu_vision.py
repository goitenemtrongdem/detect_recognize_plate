import cv2
import numpy as np
import easyocr
from datetime import datetime
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# ===============================
# Automatic Number Plate Recognition with GUI (Camera + EasyOCR)
# ===============================

class ANPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Number Plate Recognition")
        self.root.geometry("1100x600")
        self.root.resizable(False, False)

        # Variables
        self.video_stream = None
        self.running = False
        self.reader = easyocr.Reader(['en'], gpu=True)

        # ========== Layout ==========
        # Left Frame (Camera)
        self.left_frame = Frame(self.root, width=700, height=500, bg="white", relief=RIDGE, bd=2)
        self.left_frame.place(x=20, y=20)

        self.video_label = Label(self.left_frame, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)

        # Right Frame (Info)
        self.right_frame = Frame(self.root, width=340, height=500, bg="white", relief=RIDGE, bd=2)
        self.right_frame.place(x=750, y=20)

        Label(self.right_frame, text="License plate information", font=("Arial", 14, "bold"), bg="white").pack(pady=5)

        self.plate_img_label = Label(self.right_frame, bg="white", relief=SOLID, width=300, height=150)
        self.plate_img_label.pack(pady=10)

        Label(self.right_frame, text="Number Plate:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_plate = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_plate.pack(pady=2)

        Label(self.right_frame, text="Date:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_date = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_date.pack(pady=2)

        Label(self.right_frame, text="Time:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_time = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_time.pack(pady=2)

        # Buttons
        self.btn_frame = Frame(self.root, bg="white")
        self.btn_frame.place(x=400, y=540)

        Button(self.btn_frame, text="Start", font=("Arial", 12), width=10, command=self.start_camera).grid(row=0, column=0, padx=10)
        Button(self.btn_frame, text="Search", font=("Arial", 12), width=10, command=self.detect_plate).grid(row=0, column=1, padx=10)
        Button(self.btn_frame, text="Exit", font=("Arial", 12), width=10, command=self.close_app).grid(row=0, column=2, padx=10)

    # ===============================
    # Functions
    # ===============================
    def start_camera(self):
        self.video_stream = cv2.VideoCapture(0)
        self.running = True
        self.update_frame()

    def update_frame(self):
        if self.running and self.video_stream.isOpened():
            ret, frame = self.video_stream.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_frame)

    def detect_plate(self):
        if not self.video_stream or not self.video_stream.isOpened():
            return

        ret, frame = self.video_stream.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

        location = None
        for contour in cnts:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            print("Không tìm thấy vùng biển số!")
            return

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # EasyOCR
        result = self.reader.readtext(cropped_image)
        text = " ".join([res[1] for res in result]).strip()
        # --- Sau khi đã có text từ EasyOCR ---
raw_text = text  # text do EasyOCR trả về

# Hàm chuẩn hoá 2 ký tự đầu (sửa các lỗi ký tự thường gặp)
def extract_first_two_digits(s):
    if not s:
        return None
    # loại bỏ ký tự không phải alnum, replace dấu cách
    s = s.strip().upper()
    s = s.replace('-', '').replace(' ', '')
    # sửa ký tự hay bị OCR nhầm
    s = s.replace('O', '0').replace('Q', '0')   # O hoặc Q -> 0
    s = s.replace('I', '1').replace('L', '1')   # I hoặc L -> 1
    s = s.replace('Z', '2')                     # nếu cần
    s = s.replace('S', '5')                     # S -> 5 (tuỳ trường hợp)
    # lấy 2 ký tự đầu tiên là chữ số (nếu có)
    digits = ''.join([ch for ch in s if ch.isdigit()])
    if len(digits) >= 2:
        return digits[:2]
    return None

code_str = extract_first_two_digits(raw_text)

# Bảng mẫu map mã -> tỉnh (bạn nên cập nhật đầy đủ/chuẩn từ tài liệu chính thức)
plate_map = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh",
    "15": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định",
    "19": "Phú Thọ", "20": "Vĩnh Phúc", "21": "Bắc Giang",
    "22": "Bắc Ninh", "23": "Hải Dương", "24": "Hà Nam",
    "25": "Ninh Bình", "26": "Thanh Hóa", "27": "Điện Biên",
    "28": "Lai Châu", "29": "Hà Nội", "30": "Hà Nội",
    "31": "Hà Nội", "32": "Hà Nội", "33": "Hà Nội",
    "34": "Hà Nội", "35": "Hà Nội", "36": "Thanh Hóa",
    "37": "Nghệ An", "38": "Hà Tĩnh", "43": "Đà Nẵng",
    "51": "TP. Hồ Chí Minh", "52": "Bình Dương", "53": "Bình Dương",
    "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang",
    "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh",
    # ... (bổ sung nốt các mã khác theo danh sách chính thức)
}

province = None
if code_str:
    province = plate_map.get(code_str)
    if province is None:
        # thử thêm 1 lần nữa nếu có trường hợp ghép như '15'/'16' (VD: Hải Phòng là 15-16)
        # ở đây bạn có thể kiểm tra range hoặc danh sách nhiều mã cho 1 tỉnh
        province = "Không xác định (mã: {})".format(code_str)
else:
    province = "Không trích được mã tỉnh (OCR không đọc được 2 số đầu)"

# Hiển thị (ví dụ cập nhật vào entry hoặc in ra)
print("OCR raw:", raw_text)
print("Mã 2 số:", code_str)
print("Tỉnh:", province)

# Nếu muốn hiển thị ngay trên GUI:
self.txt_plate.delete(0, END)
self.txt_plate.insert(0, raw_text)
# Tạo 1 trường hiển thị tên tỉnh (nếu muốn)
try:
    self.txt_province  # nếu bạn đã tạo Entry cho province
except AttributeError:
    # tạo Entry mới tạm thời ở khung phải (nếu chưa có)
    Label(self.right_frame, text="Province:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
    self.txt_province = Entry(self.right_frame, font=("Arial", 11), width=30)
    self.txt_province.pack(pady=2)
self.txt_province.delete(0, END)
self.txt_province.insert(0, province)
        if text:
            print("Biển số nhận dạng:", text)
            self.txt_plate.delete(0, END)
            self.txt_plate.insert(0, text)

            now = datetime.now()
            self.txt_date.delete(0, END)
            self.txt_date.insert(0, now.strftime("%d/%m/%Y"))

            self.txt_time.delete(0, END)
            self.txt_time.insert(0, now.strftime("%I:%M %p"))

            # Hiển thị ảnh biển số ở khung phải
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            crop_img = Image.fromarray(cropped_rgb)
            crop_img = crop_img.resize((280, 140))
            imgtk_crop = ImageTk.PhotoImage(image=crop_img)
            self.plate_img_label.configure(image=imgtk_crop)
            self.plate_img_label.imgtk = imgtk_crop

    def close_app(self):
        self.running = False
        if self.video_stream:
            self.video_stream.release()
        self.root.destroy()

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    root = Tk()
    app = ANPR_GUI(root)
    root.mainloop()






# import cv2
# import numpy as np
# import easyocr
# from datetime import datetime
# from tkinter import *
# from tkinter import ttk
# from PIL import Image, ImageTk

# # ===============================
# # Automatic Number Plate Recognition with GUI (Camera + EasyOCR)
# # ===============================

# class ANPR_GUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Automatic Number Plate Recognition")
#         self.root.geometry("1100x600")
#         self.root.resizable(False, False)

#         # Variables
#         self.video_stream = None
#         self.running = False
#         self.reader = easyocr.Reader(['en'], gpu=True)

#         # ========== Layout ==========
#         # Left Frame (Camera)
#         self.left_frame = Frame(self.root, width=700, height=500, bg="white", relief=RIDGE, bd=2)
#         self.left_frame.place(x=20, y=20)

#         self.video_label = Label(self.left_frame, bg="black")
#         self.video_label.pack(fill=BOTH, expand=True)

#         # Right Frame (Info)
#         self.right_frame = Frame(self.root, width=340, height=500, bg="white", relief=RIDGE, bd=2)
#         self.right_frame.place(x=750, y=20)

#         Label(self.right_frame, text="License plate information", font=("Arial", 14, "bold"), bg="white").pack(pady=5)

#         self.plate_img_label = Label(self.right_frame, bg="white", relief=SOLID, width=300, height=150)
#         self.plate_img_label.pack(pady=10)

#         Label(self.right_frame, text="Number Plate:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
#         self.txt_plate = Entry(self.right_frame, font=("Arial", 11), width=30)
#         self.txt_plate.pack(pady=2)

#         Label(self.right_frame, text="Date:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
#         self.txt_date = Entry(self.right_frame, font=("Arial", 11), width=30)
#         self.txt_date.pack(pady=2)

#         Label(self.right_frame, text="Time:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
#         self.txt_time = Entry(self.right_frame, font=("Arial", 11), width=30)
#         self.txt_time.pack(pady=2)

#         # Buttons
#         self.btn_frame = Frame(self.root, bg="white")
#         self.btn_frame.place(x=400, y=540)

#         Button(self.btn_frame, text="Start", font=("Arial", 12), width=10, command=self.start_camera).grid(row=0, column=0, padx=10)
#         Button(self.btn_frame, text="Search", font=("Arial", 12), width=10, command=self.detect_plate).grid(row=0, column=1, padx=10)
#         Button(self.btn_frame, text="Exit", font=("Arial", 12), width=10, command=self.close_app).grid(row=0, column=2, padx=10)

#     # ===============================
#     # Functions
#     # ===============================
#     def start_camera(self):
#         self.video_stream = cv2.VideoCapture(0)
#         self.running = True
#         self.update_frame()

#     def update_frame(self):
#         if self.running and self.video_stream.isOpened():
#             ret, frame = self.video_stream.read()
#             if ret:
#                 frame = cv2.flip(frame, 1)
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(rgb)
#                 imgtk = ImageTk.PhotoImage(image=img)
#                 self.video_label.imgtk = imgtk
#                 self.video_label.configure(image=imgtk)
#             self.root.after(10, self.update_frame)

#     def detect_plate(self):
#         if not self.video_stream or not self.video_stream.isOpened():
#             return

#         ret, frame = self.video_stream.read()
#         if not ret:
#             return

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.bilateralFilter(gray, 11, 17, 17)
#         edged = cv2.Canny(gray, 30, 200)

#         # ===============================
#         # --- Cải tiến phát hiện biển số ---
#         # ===============================
#         cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]

#         location = None
#         for contour in cnts:
#             area = cv2.contourArea(contour)
#             if area < 500 or area > 60000:
#                 continue

#             rect = cv2.minAreaRect(contour)
#             box = cv2.boxPoints(rect)
#             box = np.intp(box)

#             w, h = rect[1]
#             if w == 0 or h == 0:
#                 continue
#             ratio = max(w, h) / min(w, h)

#             # Kiểm tra tỷ lệ khung hình hợp lý (biển số Việt Nam khoảng 2:1 -> 6:1)
#             if 2 <= ratio <= 6:
#                 location = box
#                 break

#         if location is None:
#             print("Không tìm thấy vùng biển số!")
#             return

#         # Vẽ contour quanh vùng biển số
#         detected = frame.copy()
#         cv2.drawContours(detected, [location], -1, (0, 255, 0), 2)
#         cv2.imshow("Detected Plate Contour", detected)
#         cv2.waitKey(1)

#         # Mask và crop vùng biển số
#         mask = np.zeros(gray.shape, np.uint8)
#         cv2.drawContours(mask, [location], 0, 255, -1)

#         (x, y) = np.where(mask == 255)
#         (x1, y1) = (np.min(x), np.min(y))
#         (x2, y2) = (np.max(x), np.max(y))
#         cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

#         # ===============================
#         # --- EasyOCR đọc biển số ---
#         # ===============================
#         result = self.reader.readtext(cropped_image)
#         text = " ".join([res[1] for res in result]).strip()

#         if text:
#             print("Biển số nhận dạng:", text)
#             self.txt_plate.delete(0, END)
#             self.txt_plate.insert(0, text)

#             now = datetime.now()
#             self.txt_date.delete(0, END)
#             self.txt_date.insert(0, now.strftime("%d/%m/%Y"))

#             self.txt_time.delete(0, END)
#             self.txt_time.insert(0, now.strftime("%I:%M %p"))

#             # Hiển thị ảnh biển số ở khung phải
#             cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
#             crop_img = Image.fromarray(cropped_rgb)
#             crop_img = crop_img.resize((280, 140))
#             imgtk_crop = ImageTk.PhotoImage(image=crop_img)
#             self.plate_img_label.configure(image=imgtk_crop)
#             self.plate_img_label.imgtk = imgtk_crop

#     def close_app(self):
#         self.running = False
#         if self.video_stream:
#             self.video_stream.release()
#         # cv2.destroyAllWindows()
#         self.root.destroy()

# # ===============================
# # Run App
# # ===============================
# if __name__ == "__main__":
#     root = Tk()
#     app = ANPR_GUI(root)
#     root.mainloop()
