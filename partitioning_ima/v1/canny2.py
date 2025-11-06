# """
# canny2.py
# Phiên bản: Canny + Distance Transform + Watershed
# GUI: Tkinter hiển thị 6 ảnh (2x3) tương ứng từng bước:
# 1. Ảnh gốc
# 2. Cạnh Canny
# 3. Ảnh nghịch đảo (sơ bộ vùng vật thể)
# 4. Bản đồ khoảng cách (distance transform)
# 5. Markers
# 6. Kết quả phân vùng

# Chạy: python canny_watershed_gui.py
# """

# import cv2
# import numpy as np
# from tkinter import *
# from tkinter import filedialog
# from PIL import Image, ImageTk
# from skimage import color, morphology, measure, segmentation, util
# from scipy import ndimage as ndi

# # --- Cấu hình hiển thị ---
# THUMB_W, THUMB_H = 360, 240  # kích thước mỗi ô ảnh

# # --- Hàm tiện ích: convert OpenCV image -> ImageTk ---
# def cv_to_ImageTk(cv_img, thumb_w=THUMB_W, thumb_h=THUMB_H, gray=False):
#     if gray:
#         pil = Image.fromarray(cv_img).convert("L")
#     else:
#         if cv_img.ndim == 2:
#             cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
#         pil = Image.fromarray(cv_img).convert("RGB")

#     pil.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
#     return ImageTk.PhotoImage(pil)


# # --- Pipeline xử lý ảnh ---
# def process_pipeline(path, low_thresh=50, high_thresh=150, min_obj_area=500):
#     # 1️⃣ Đọc ảnh
#     orig_bgr = cv2.imread(path)
#     if orig_bgr is None:
#         raise ValueError("Không đọc được ảnh.")
#     orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

#     # 2️⃣ Làm mịn & chuyển xám
#     gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 2)

#     # 3️⃣ Tách biên Canny
#     edges = cv2.Canny(blurred, low_thresh, high_thresh)

#     # 4️⃣ Nghịch đảo biên để lấy vùng vật thể sơ bộ
#     inv = cv2.bitwise_not(edges)
#     _, inv_bin = cv2.threshold(inv, 127, 255, cv2.THRESH_BINARY)

#     # 5️⃣ Loại bỏ nhiễu nhỏ
#     inv_bool = inv_bin.astype(bool)
#     inv_clean = morphology.remove_small_objects(inv_bool, min_size=min_obj_area, connectivity=2)
#     inv_clean = util.img_as_ubyte(inv_clean)

#     # 6️⃣ Distance transform
#     distance = ndi.distance_transform_edt(inv_clean > 0)
#     if distance.max() > 0:
#         distance_norm = (distance / distance.max() * 255).astype(np.uint8)
#     else:
#         distance_norm = (distance * 255).astype(np.uint8)

#     # 7️⃣ Tìm markers (cực đại nội bộ)
#     local_max = morphology.local_maxima(distance)
#     local_max_clean = morphology.remove_small_objects(local_max, min_size=20, connectivity=2)
#     markers = measure.label(local_max_clean)

#     # 8️⃣ Watershed
#     mask = inv_clean > 0
#     labels_ws = segmentation.watershed(-distance, markers, mask=mask)

#     # 9️⃣ Lọc bỏ vùng nhỏ
#     props = measure.regionprops(labels_ws)
#     labels_filtered = np.zeros_like(labels_ws)
#     label_id = 1
#     for prop in props:
#         if prop.area >= min_obj_area:
#             labels_filtered[labels_ws == prop.label] = label_id
#             label_id += 1

#     # 🔟 Tô màu vùng kết quả
#     colored = color.label2rgb(labels_filtered, bg_label=0, bg_color=(0, 0, 0), kind='overlay')
#     colored_uint8 = (np.clip(colored, 0, 1) * 255).astype(np.uint8)

#     # Tạo ảnh markers màu để hiển thị
#     markers_viz = color.label2rgb(markers, bg_label=0, bg_color=(0, 0, 0), kind='overlay')
#     markers_viz = (np.clip(markers_viz, 0, 1) * 255).astype(np.uint8)

#     num_regions = int(labels_filtered.max())

#     return {
#         'orig_rgb': orig_rgb,
#         'edges': edges,
#         'inv_clean': inv_clean,
#         'distance_norm': distance_norm,
#         'markers_viz': markers_viz,
#         'colored': colored_uint8,
#         'num_regions': num_regions
#     }


# # --- GIAO DIỆN GUI ---
# class App:
#     def __init__(self, master):
#         self.master = master
#         master.title("Canny + Watershed - Phân vùng ảnh")
#         master.configure(bg="#ececec")

#         self.frames = []
#         self.labels = []
#         self.photos = [None] * 6

#         titles = [
#             "Ảnh gốc",
#             "Cạnh Canny",
#             "Vùng sơ bộ (nghịch đảo)",
#             "Distance map",
#             "Markers",
#             "Kết quả phân vùng"
#         ]

#         for r in range(2):
#             for c in range(3):
#                 idx = r * 3 + c
#                 f = Frame(master, width=THUMB_W, height=THUMB_H + 24, bd=1, relief="sunken", bg="#f8f8f8")
#                 f.grid(row=r, column=c, padx=6, pady=6)
#                 f.grid_propagate(False)
#                 Label(f, text=titles[idx], bg="#f8f8f8", font=("Arial", 10, "bold")).pack(side="top", pady=2)
#                 canvas = Label(f, bg="#ddd")
#                 canvas.pack(expand=True)
#                 self.frames.append(f)
#                 self.labels.append(canvas)

#         # Nút chọn ảnh
#         btn_frame = Frame(master, bg="#ececec")
#         btn_frame.grid(row=2, column=0, columnspan=3, pady=10)
#         self.btn = Button(btn_frame, text="Chọn ảnh và xử lý", command=self.open_and_process, padx=10, pady=6, bg="#4287f5", fg="white", font=("Arial", 10, "bold"))
#         self.btn.pack(side="left", padx=10)

#         self.info_label = Label(btn_frame, text="Số vùng: -", bg="#ececec", font=("Arial", 11, "bold"))
#         self.info_label.pack(side="left", padx=20)

#     def open_and_process(self):
#         path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.bmp;*.tif;*.tiff")])
#         if not path:
#             return
#         try:
#             res = process_pipeline(path)
#         except Exception as e:
#             self.info_label.config(text=f"Lỗi: {e}")
#             return

#         imgs = [
#             (res['orig_rgb'], False),
#             (res['edges'], True),
#             (res['inv_clean'], True),
#             (res['distance_norm'], True),
#             (res['markers_viz'], False),
#             (res['colored'], False)
#         ]

#         for i, (img, gray) in enumerate(imgs):
#             photo = cv_to_ImageTk(img, THUMB_W, THUMB_H, gray=gray)
#             self.photos[i] = photo
#             self.labels[i].config(image=photo)

#         self.info_label.config(text=f"Số vùng phát hiện: {res['num_regions']}")


# if __name__ == "__main__":
#     root = Tk()
#     app = App(root)
#     root.resizable(False, False)
#     root.mainloop()







import cv2
import numpy as np
from skimage import morphology, measure, segmentation, color, exposure, util
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==============================
# Xử lý pipeline
# ==============================
def process_pipeline(path, low_thresh=50, high_thresh=150, min_obj_area=300):
    # 1️⃣ Đọc và chuyển sang grayscale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Tách biên Canny
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # 3️⃣ Nghịch đảo để làm vùng sơ bộ
    inv_bin = cv2.bitwise_not(edges)

    # 4️⃣ 🌈 Tăng độ tương phản
    inv_bin_eq = exposure.equalize_hist(inv_bin)  # normalize contrast
    inv_bin_eq = (inv_bin_eq * 255).astype(np.uint8)

    # 5️⃣ 🧹 Lọc nhiễu (remove small objects)
    bw_bool = inv_bin_eq > 128
    bw_clean = morphology.remove_small_objects(bw_bool, min_size=min_obj_area)
    bw_clean = util.img_as_ubyte(bw_clean)

    # 6️⃣ Distance map
    distance = cv2.distanceTransform(bw_clean, cv2.DIST_L2, 5)
    distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 7️⃣ Marker để phân vùng
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)

    # 8️⃣ Phân vùng bằng watershed
    labels_ws = segmentation.watershed(-distance, markers, mask=bw_clean)
    result_overlay = color.label2rgb(labels_ws, bg_label=0)

    # 9️⃣ Đếm số vùng
    num_regions = np.max(labels_ws)

    return img, edges, inv_bin_eq, distance_norm, markers, result_overlay, num_regions

# ==============================
# Giao diện GUI Tkinter
# ==============================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Canny + Tăng tương phản + Watershed - Phân vùng ảnh")

        self.images = [None] * 6
        self.labels = []

        titles = ["Ảnh gốc", "Cạnh Canny", "Vùng sơ bộ (nghịch đảo)",
                  "Tăng tương phản + lọc nhiễu", "Markers", "Kết quả phân vùng"]

        # 2 hàng, 3 cột
        for i in range(6):
            frame = Frame(root, bd=2, relief="groove")
            frame.grid(row=i // 3, column=i % 3, padx=5, pady=5)
            Label(frame, text=titles[i], font=("Arial", 10, "bold")).pack()
            lbl = Label(frame)
            lbl.pack()
            self.labels.append(lbl)

        # Nút chọn ảnh
        self.btn = Button(root, text="Chọn ảnh và xử lý", command=self.load_image,
                          bg="#0078D7", fg="white", font=("Arial", 10, "bold"))
        self.btn.grid(row=3, column=0, columnspan=3, pady=10)

        self.result_label = Label(root, text="Số vùng phát hiện: 0",
                                  font=("Arial", 11, "bold"))
        self.result_label.grid(row=4, column=0, columnspan=3)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return

        try:
            img, edges, inv_bin_eq, distance_norm, markers, result_overlay, num_regions = process_pipeline(path)

            # Chuyển markers sang dạng hiển thị được
            markers_vis = (markers.astype(np.float32) / np.max(markers) * 255).astype(np.uint8)

            imgs = [img, edges, inv_bin_eq, distance_norm, markers_vis, result_overlay]

            for i, im in enumerate(imgs):
                if len(im.shape) == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                im = cv2.resize(im, (250, 200))
                im = ImageTk.PhotoImage(Image.fromarray(im))
                self.labels[i].config(image=im)
                self.labels[i].image = im

            self.result_label.config(text=f"Số vùng phát hiện: {num_regions}")

        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

# ==============================
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
