import cv2
import os
import datetime
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(camera_index=0, save_dir="captured"):
    """
    Pipeline xử lý ảnh:
    1. Chụp ảnh từ camera
    2. Chuyển sang ảnh xám
    3. Tăng tương phản (TopHat + Gray - BlackHat)
    4. Giảm nhiễu Gaussian
    5. Nhị phân hóa ngưỡng động Gaussian
    6. Phát hiện cạnh bằng Canny
    7. Tìm và vẽ contour (Suzuki’s Tracing)
    """
    ensure_dir(save_dir)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Không thể mở camera index={camera_index}.")
        return

    print("✅ Đã mở camera. Nhấn 's' để lưu ảnh cạnh, 'q' để thoát.")

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không nhận được khung ảnh.")
            break

        # --- 1. Chuyển ảnh sang xám ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 2. Tăng tương phản (TopHat + BlackHat) ---
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        # --- 3. Giảm nhiễu Gaussian ---
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # --- 4. Nhị phân hóa ngưỡng động Gaussian ---
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 2
        )

        # --- 5. Phát hiện cạnh bằng Canny ---
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

        # --- 6. Tìm và vẽ contour (Suzuki’s Tracing) ---
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 20000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Lọc vùng nghi ngờ là biển số
            if 2.0 < aspect_ratio < 6.0:
                # Vẽ contour chính xác (màu hồng)
                cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)

                # (Tùy chọn) Vẽ khung phụ để tham chiếu
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

                # Ghi chú vị trí contour
                cv2.putText(frame, f"Contour {i}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # --- 7. Hiển thị ---
        cv2.imshow("Gray", gray)
        cv2.imshow("Enhanced", enhanced)
        cv2.imshow("Adaptive Threshold", binary)
        cv2.imshow("Edges (Canny)", edges)
        cv2.imshow("Detected Plates (Contours)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(save_dir, f"edges_{timestamp}.png")
            cv2.imwrite(filename, edges)
            print(f"💾 Đã lưu ảnh cạnh: {filename}")
        elif key == ord('q') or key == 27:
            print("👋 Thoát chương trình.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0, save_dir="captured")
