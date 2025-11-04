from ultralytics import YOLO
import cv2
import os
import numpy as np
import easyocr

# 1️⃣ Nạp mô hình YOLO
model = YOLO("runs/detect/license_train2/weights/best.pt")

# 2️⃣ Đường dẫn ảnh đầu vào
source = r"C:\Users\Admin\Downloads\new_ima\2.jpg"

# 3️⃣ Đọc ảnh
img = cv2.imread(source)
if img is None:
    raise ValueError(f"Không thể đọc ảnh tại đường dẫn: {source}")

# 4️⃣ Lọc nhiễu
denoised = cv2.GaussianBlur(img, (5, 5), 0)

# 5️⃣ Nhận diện biển số bằng YOLO
results = model(denoised, conf=0.5, verbose=False)

# 6️⃣ Tạo thư mục lưu kết quả chính
base_dir = "output_steps"
os.makedirs(base_dir, exist_ok=True)

# 7️⃣ Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# 8️⃣ Xử lý từng biển số phát hiện được
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        continue

    # 🔹 Tạo thư mục riêng cho mỗi biển số
    plate_dir = os.path.join(base_dir, f"plate_{i+1}")
    os.makedirs(plate_dir, exist_ok=True)

    # 🔸 Lưu ảnh gốc
    original_path = os.path.join(plate_dir, "step1_original.jpg")
    cv2.imwrite(original_path, cropped)

    # --- Chuyển sang ảnh xám
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(plate_dir, "step2_gray.jpg")
    cv2.imwrite(gray_path, gray)

    # --- Nhị phân hóa (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_path = os.path.join(plate_dir, "step3_binary.jpg")
    cv2.imwrite(binary_path, binary)

    # 🔸 Morphology để tách ký tự dính
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    morph_path = os.path.join(plate_dir, "step4_morphology.jpg")
    cv2.imwrite(morph_path, binary)

    # 8️⃣ Tìm contour cho ký tự
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    h_plate, w_plate = binary.shape[:2]
    plate_area = h_plate * w_plate

    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0
        area_ratio = area / plate_area

        if (
            0.1 < aspect_ratio < 1.2
            and 0.005 < area_ratio < 0.2
            and h > 0.25 * h_plate
        ):
            valid_contours.append((x, y, w, h))

    if not valid_contours:
        print(f"⚠️ Biển số {i+1}: không tìm thấy ký tự nào!")
        continue

    # 🔹 Hàm sắp xếp ký tự theo hàng ngang & dọc
    def sort_by_rows(boxes, y_thresh=10):
        rows = []
        for b in sorted(boxes, key=lambda b: b[1]):
            x, y, w, h = b
            found_row = False
            for row in rows:
                if abs(row[0][1] - y) < y_thresh:
                    row.append(b)
                    found_row = True
                    break
            if not found_row:
                rows.append([b])
        for r in rows:
            r.sort(key=lambda b: b[0])
        return [b for row in rows for b in row]

    ordered_contours = sort_by_rows(valid_contours, y_thresh=12)

    # 9️⃣ Tạo thư mục lưu ký tự
    char_dir = os.path.join(plate_dir, "chars")
    os.makedirs(char_dir, exist_ok=True)

    recognized_text = ""

    # 🔹 Cắt & lưu từng ký tự riêng biệt + nhận diện
    for idx, (x, y, w, h) in enumerate(ordered_contours, start=1):
        char_crop = binary[y:y + h, x:x + w]
        char_resized = cv2.resize(char_crop, (50, 80))
        char_path = os.path.join(char_dir, f"char_{idx}.jpg")
        cv2.imwrite(char_path, char_resized)

        # ⚙️ Đọc lại ảnh ký tự và chuyển sang xám trước khi OCR
        char_img = cv2.imread(char_path, cv2.IMREAD_GRAYSCALE)
        if char_img is None:
            print(f"⚠️ Không thể đọc ký tự {char_path}")
            continue

        result = reader.readtext(char_img, detail=0, paragraph=False)
        if result:
            recognized_text += result[0]

        # Vẽ khung vàng lên ký tự
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(contour_img, str(idx), (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 🔹 Ghi chuỗi biển số vào file và ảnh
    text_path = os.path.join(plate_dir, "recognized_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(recognized_text)

    cv2.putText(contour_img, recognized_text, (10, h_plate - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 🔹 Lưu ảnh có contour vàng + text
    final_path = os.path.join(plate_dir, "step6_final_with_text.jpg")
    cv2.imwrite(final_path, contour_img)

    print(f"✅ Biển số {i+1}: {recognized_text}")
    print(f"📝 Lưu vào: {text_path}")

try:
    cv2.destroyAllWindows()
except:
    pass
print("\n🎯 Hoàn tất toàn bộ xử lý và OCR! Kết quả trong thư mục output_steps/")
