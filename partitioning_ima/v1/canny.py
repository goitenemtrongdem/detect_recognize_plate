import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from skimage import filters, morphology, measure, color
import matplotlib.pyplot as plt

# ==========================
# 📌 HÀM CHÍNH XỬ LÝ ẢNH
# ==========================
def xu_ly_anh():
    # --- Chọn ảnh ---
    duong_dan = filedialog.askopenfilename(
        filetypes=[("Tất cả ảnh", "*.jpg;*.png;*.bmp;*.tif")]
    )
    if not duong_dan:
        print("Đã hủy.")
        return

    # --- Đọc ảnh ---
    anh_goc = cv2.imread(duong_dan)
    anh_goc_rgb = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2RGB)

    # --- Chuyển ảnh xám ---
    anh_xam = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2GRAY)

    # --- Làm mịn (Gaussian Blur) ---
    anh_min = cv2.GaussianBlur(anh_xam, (5, 5), 2)

    # --- Phát hiện biên Canny ---
    nguong_thap, nguong_cao = 80, 200
    anh_canh = cv2.Canny(anh_min, nguong_thap, nguong_cao)

    # --- Giãn nở ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    anh_dan_no = cv2.dilate(anh_canh, kernel)

    # --- Lấp đầy vùng ---
    anh_lap_day = anh_dan_no.copy()
    contours, _ = cv2.findContours(anh_lap_day, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(anh_lap_day, [cnt], 0, 255, -1)

    # --- Lọc bỏ vùng nhỏ ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anh_lap_day)
    dien_tich_toi_thieu = 1000
    anh_loc = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):  # bỏ nền
        if stats[i, cv2.CC_STAT_AREA] >= dien_tich_toi_thieu:
            anh_loc[labels == i] = 255

    # --- Gán nhãn và tô màu ---
    nhan = measure.label(anh_loc > 0, connectivity=2)
    anh_mau = color.label2rgb(nhan, bg_label=0, bg_color=(0, 0, 0), kind='overlay')

    # --- Hiển thị toàn bộ kết quả ---
    hienthi_ketqua(anh_goc_rgb, anh_canh, anh_dan_no, anh_lap_day, anh_loc, anh_mau, len(np.unique(nhan))-1)


# ==========================
# 📊 HIỂN THỊ KẾT QUẢ BẰNG MATPLOTLIB
# ==========================
def hienthi_ketqua(anh_goc, anh_canh, anh_dan_no, anh_lap_day, anh_loc, anh_mau, so_vung):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Phân vùng ảnh Canny - Python", fontsize=14)

    axs[0, 0].imshow(anh_goc)
    axs[0, 0].set_title("Ảnh gốc")

    axs[0, 1].imshow(anh_canh, cmap='gray')
    axs[0, 1].set_title("Cạnh Canny")

    axs[0, 2].imshow(anh_dan_no, cmap='gray')
    axs[0, 2].set_title("Giãn nở cạnh")

    axs[1, 0].imshow(anh_lap_day, cmap='gray')
    axs[1, 0].set_title("Lấp vùng")

    axs[1, 1].imshow(anh_loc, cmap='gray')
    axs[1, 1].set_title("Lọc nhiễu")

    axs[1, 2].imshow(anh_mau)
    axs[1, 2].set_title("Kết quả phân vùng")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    print(f"Tìm thấy {so_vung} vùng đối tượng.")


# ==========================
# 🖥️ GIAO DIỆN TKINTER
# ==========================
root = Tk()
root.title("Phân vùng ảnh Canny - Python GUI")
root.geometry("400x200")
root.configure(bg="#ECECEC")

Label(root, text="Phân vùng ảnh dùng Canny", font=("Arial", 16, "bold"), bg="#ECECEC").pack(pady=20)

btn = Button(root, text="Chọn ảnh và xử lý", font=("Arial", 12), command=xu_ly_anh, bg="#0078D7", fg="white", padx=10, pady=5)
btn.pack(pady=10)

root.mainloop()
