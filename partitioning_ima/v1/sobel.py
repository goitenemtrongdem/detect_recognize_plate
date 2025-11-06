import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from skimage import measure, color
import matplotlib.pyplot as plt

# ==========================
# 📌 HÀM CHÍNH XỬ LÝ ẢNH (SOBEL)
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

    # --- Cân bằng histogram cục bộ (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    anh_eq = clahe.apply(anh_xam)

    # --- Làm mịn (Gaussian Blur) ---
    anh_min = cv2.GaussianBlur(anh_eq, (5, 5), 1.5)

    # --- Tính gradient Sobel ---
    sobelx = cv2.Sobel(anh_min, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(anh_min, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # --- Nhị phân hóa bằng Otsu ---
    _, anh_canh = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Morphology: làm kín và loại nhiễu ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    anh_dan_no = cv2.morphologyEx(anh_canh, cv2.MORPH_CLOSE, kernel, iterations=2)
    anh_dan_no = cv2.morphologyEx(anh_dan_no, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Lấp vùng bên trong ---
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

    # --- Hiển thị kết quả ---
    hienthi_ketqua(
        anh_goc_rgb, magnitude, anh_dan_no, anh_lap_day, anh_loc, anh_mau, len(np.unique(nhan)) - 1
    )


# ==========================
# 📊 HIỂN THỊ KẾT QUẢ BẰNG MATPLOTLIB
# ==========================
def hienthi_ketqua(anh_goc, anh_canh, anh_dan_no, anh_lap_day, anh_loc, anh_mau, so_vung):
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Phân vùng ảnh Sobel - Python", fontsize=14)

    axs[0, 0].imshow(anh_goc)
    axs[0, 0].set_title("Ảnh gốc")

    axs[0, 1].imshow(anh_canh, cmap='gray')
    axs[0, 1].set_title("Biên Sobel (magnitude)")

    axs[0, 2].imshow(anh_dan_no, cmap='gray')
    axs[0, 2].set_title("Sau Morphology")

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
root.title("Phân vùng ảnh Sobel - Python GUI")
root.geometry("400x200")
root.configure(bg="#ECECEC")

Label(
    root,
    text="Phân vùng ảnh dùng Sobel",
    font=("Arial", 16, "bold"),
    bg="#ECECEC",
).pack(pady=20)

btn = Button(
    root,
    text="📂 Chọn ảnh và xử lý",
    font=("Arial", 12),
    command=xu_ly_anh,
    bg="#0078D7",
    fg="white",
    padx=10,
    pady=5,
)
btn.pack(pady=10)

root.mainloop()
