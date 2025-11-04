# ============================================================
# 🧠 TRAIN YOLOv8 MODEL - DÙNG GPU (CUDA)
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import multiprocessing

def main():
    # 🔹 1. Chọn mô hình YOLOv8 (nhỏ, nhanh, dùng cho thử nghiệm)
    model = YOLO("yolov8n.pt")  # có thể đổi thành yolov8s.pt, yolov8m.pt nếu GPU mạnh hơn

    # 🔹 2. Đường dẫn đến file cấu hình dataset (data.yaml do bạn export từ Roboflow)
    data_path = "data.yaml"  # thay bằng đường dẫn thực tế, ví dụ: r"C:\Users\Admin\Downloads\data.yaml"

    # 🔹 3. Huấn luyện mô hình
    results = model.train(
        data=data_path,      # file yaml
        epochs=100,          # số vòng huấn luyện (nên 100 nếu dữ liệu ít)
        imgsz=640,           # kích thước ảnh
        batch=8,             # số ảnh mỗi lần huấn luyện (tùy VRAM GPU)
        device=0,            # dùng GPU số 0 (CPU thì để 'cpu')
        workers=4,           # số luồng xử lý dữ liệu
        name="license_train",# tên folder lưu kết quả (trong runs/detect/)
        patience=20,         # dừng sớm nếu model không cải thiện
    )

    print("✅ Training hoàn tất!")
    print("📁 Model lưu tại:", results.save_dir)

# ============================================================
# ⚙️ Đảm bảo Windows không lỗi multiprocessing
# ============================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
