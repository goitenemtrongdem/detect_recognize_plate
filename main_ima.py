import cv2
import numpy as np
import easyocr
from datetime import datetime
from tkinter import *
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# ===============================
# Automatic Number Plate Recognition with GUI + Province detection
# ===============================

class ANPR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Automatic Number Plate Recognition (Vietnam)")
        self.root.geometry("1100x620")
        self.root.resizable(False, False)

        # Variables
        self.loaded_image = None   # <--- Ảnh đã chọn
        self.reader = easyocr.Reader(['en'], gpu=True)

        # =======================
        # Layout
        # =======================
        # Left Frame (Image display)
        self.left_frame = Frame(self.root, width=700, height=500, bg="white", relief=RIDGE, bd=2)
        self.left_frame.place(x=20, y=20)

        self.video_label = Label(self.left_frame, bg="black")
        self.video_label.pack(fill=BOTH, expand=True)

        # Right Frame (Information)
        self.right_frame = Frame(self.root, width=340, height=500, bg="white", relief=RIDGE, bd=2)
        self.right_frame.place(x=750, y=20)

        Label(self.right_frame, text="License Plate Information", font=("Arial", 14, "bold"), bg="white").pack(pady=5)

        self.plate_img_label = Label(self.right_frame, bg="white", relief=SOLID, width=300, height=150)
        self.plate_img_label.pack(pady=10)

        # --- Information Fields ---
        Label(self.right_frame, text="Number Plate:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_plate = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_plate.pack(pady=2)

        Label(self.right_frame, text="Province:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_province = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_province.pack(pady=2)

        Label(self.right_frame, text="Date:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_date = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_date.pack(pady=2)

        Label(self.right_frame, text="Time:", font=("Arial", 11, "bold"), bg="white").pack(anchor=W, padx=10)
        self.txt_time = Entry(self.right_frame, font=("Arial", 11), width=30)
        self.txt_time.pack(pady=2)

        # Buttons
        self.btn_frame = Frame(self.root, bg="white")
        self.btn_frame.place(x=400, y=540)

        Button(self.btn_frame, text="Open Image", font=("Arial", 12), width=12, command=self.open_image).grid(row=0, column=0, padx=10)
        Button(self.btn_frame, text="Detect", font=("Arial", 12), width=10, command=self.detect_plate).grid(row=0, column=1, padx=10)
        Button(self.btn_frame, text="Exit", font=("Arial", 12), width=10, command=self.close_app).grid(row=0, column=2, padx=10)

        # Bảng mã tỉnh
        self.plate_map = {
            "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh",
            "15": "Hải Phòng", "16": "Hải Phòng", "17": "Thái Bình",
            "18": "Nam Định", "19": "Phú Thọ", "20": "Vĩnh Phúc",
            "21": "Bắc Giang", "22": "Bắc Ninh", "23": "Hải Dương",
            "24": "Hà Nam", "25": "Ninh Bình", "26": "Thanh Hóa",
            "27": "Điện Biên", "28": "Lai Châu", "29": "Hà Nội",
            "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội",
            "33": "Hà Nội", "34": "Hà Nội", "35": "Hà Nội",
            "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh",
            "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông",
            "49": "Lâm Đồng", "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh",
            "53": "Bình Dương", "54": "TP. Hồ Chí Minh", "56": "Bà Rịa - Vũng Tàu",
            "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang",
            "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh", "61": "Bình Dương",
            "99": "Bắc Ninh", "95" : "Hậu Giang"
        }

    # =======================
    # Open Image Instead of Camera
    # =======================
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        self.loaded_image = cv2.imread(file_path)
        if self.loaded_image is None:
            return

        # Hiển thị ảnh lên khung
        rgb = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_pil = img_pil.resize((700, 500))
        imgtk = ImageTk.PhotoImage(img_pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # =======================
    # Plate Detection + Province Mapping
    # =======================
    def detect_plate(self):
        if self.loaded_image is None:
            print("Chưa chọn ảnh!")
            return

        frame = self.loaded_image.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 15, 25, 25)
        edged = cv2.Canny(gray, 20, 120)

        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        location = None
        for contour in cnts:
            epsilon = 0.01 * cv2.arcLength(contour, True)
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

        if text:
            print("Biển số nhận dạng:", text)
            self.txt_plate.delete(0, END)
            self.txt_plate.insert(0, text)

            # --- Extract province ---
            province = self.detect_province(text)
            self.txt_province.delete(0, END)
            self.txt_province.insert(0, province)

            # --- Date & Time ---
            now = datetime.now()
            self.txt_date.delete(0, END)
            self.txt_date.insert(0, now.strftime("%d/%m/%Y"))

            self.txt_time.delete(0, END)
            self.txt_time.insert(0, now.strftime("%I:%M %p"))

            # --- Display cropped plate image ---
            cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            crop_img = Image.fromarray(cropped_rgb)
            crop_img = crop_img.resize((280, 140))
            imgtk_crop = ImageTk.PhotoImage(image=crop_img)
            self.plate_img_label.configure(image=imgtk_crop)
            self.plate_img_label.imgtk = imgtk_crop

    # =======================
    # Helper: Extract Province Code
    # =======================
    def detect_province(self, raw_text):
        if not raw_text:
            return "Không trích được mã tỉnh"

        s = raw_text.strip().upper()
        s = s.replace('-', '').replace(' ', '')
        s = s.replace('O', '0').replace('Q', '0')
        s = s.replace('I', '1').replace('L', '1').replace('Z', '2').replace('S', '5').replace('B', '3')
        digits = ''.join([ch for ch in s if ch.isdigit()])

        if len(digits) < 2:
            return "Không xác định (OCR lỗi)"

        code = digits[:2]
        province = self.plate_map.get(code, f"Không xác định (mã {code})")
        print(f"Mã: {code} → Tỉnh: {province}")
        return province

    def close_app(self):
        self.root.destroy()


# =======================
# Run App
# =======================
if __name__ == "__main__":
    root = Tk()
    app = ANPR_GUI(root)
    root.mainloop()
