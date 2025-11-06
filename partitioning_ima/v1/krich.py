import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# ======================================
# Kirsch Edge Detection + Color Segmentation GUI
# ======================================

class KirschSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kirsch Edge Segmentation with Color Regions")
        self.root.geometry("1150x750")
        self.root.configure(bg="#f4f4f4")

        self.images = []
        self.titles = []
        self.index = 0

        # ---------- GUI Layout ----------
        Label(self.root, text="🎨 Kirsch Edge-Based Image Segmentation",
              font=("Helvetica", 17, "bold"), bg="#f4f4f4", fg="#333").pack(pady=10)

        self.status_label = Label(self.root, text="👉 Hãy chọn ảnh để bắt đầu",
                                  font=("Helvetica", 12), bg="#f4f4f4", fg="gray")
        self.status_label.pack(pady=5)

        frame_btn = Frame(self.root, bg="#f4f4f4")
        frame_btn.pack(pady=10)

        Button(frame_btn, text="📂 Chọn ảnh", command=self.load_image,
               font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=15).grid(row=0, column=0, padx=10)
        Button(frame_btn, text="⏮ Quay lại", command=self.prev_step,
               font=("Helvetica", 12), width=15).grid(row=0, column=1, padx=10)
        Button(frame_btn, text="▶ Tiếp theo", command=self.next_step,
               font=("Helvetica", 12), width=15).grid(row=0, column=2, padx=10)

        # khung hiển thị ảnh
        self.img_label = Label(self.root, bg="#ddd", relief="ridge", width=800, height=500)
        self.img_label.pack(pady=15)

        self.step_label = Label(self.root, text="", font=("Helvetica", 13, "bold"),
                                bg="#f4f4f4", fg="#222")
        self.step_label.pack()

    # ---------- Kirsch Edge Operator ----------
    def kirsch_edge_detection(self, gray):
        kirsch_masks = [
            np.array([[5, 5, 5],
                      [-3, 0, -3],
                      [-3, -3, -3]]),  # North
            np.array([[5, 5, -3],
                      [5, 0, -3],
                      [-3, -3, -3]]),  # North-East
            np.array([[5, -3, -3],
                      [5, 0, -3],
                      [5, -3, -3]]),  # East
            np.array([[-3, -3, -3],
                      [5, 0, -3],
                      [5, 5, -3]]),   # South-East
            np.array([[-3, -3, -3],
                      [-3, 0, -3],
                      [5, 5, 5]]),   # South
            np.array([[-3, -3, -3],
                      [-3, 0, 5],
                      [-3, 5, 5]]),  # South-West
            np.array([[-3, -3, 5],
                      [-3, 0, 5],
                      [-3, -3, 5]]), # West
            np.array([[-3, 5, 5],
                      [-3, 0, 5],
                      [-3, -3, -3]]) # North-West
        ]

        edges = [cv2.filter2D(gray, cv2.CV_64F, mask) for mask in kirsch_masks]
        kirsch_combined = np.max(edges, axis=0)
        kirsch_combined = cv2.convertScaleAbs(kirsch_combined)
        return kirsch_combined

    # ---------- Pipeline xử lý ----------
    def process_pipeline(self, path):
        self.images, self.titles = [], []

        # B1: Đọc ảnh
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.images.append(img_rgb)
        self.titles.append("Ảnh gốc")

        # B2: Chuyển xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        self.titles.append("Ảnh xám")

        # B3: Làm mượt
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self.images.append(cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB))
        self.titles.append("Làm mượt Gaussian")

        # B4: Tách biên Kirsch
        kirsch = self.kirsch_edge_detection(blur)
        self.images.append(cv2.cvtColor(kirsch, cv2.COLOR_GRAY2RGB))
        self.titles.append("Biên Kirsch (8 hướng)")

        # B5: Ngưỡng hóa
        _, binary = cv2.threshold(kirsch, 100, 255, cv2.THRESH_BINARY)
        self.images.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
        self.titles.append("Nhị phân ảnh biên")

        # B6: Morph Closing
        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.images.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB))
        self.titles.append("Đóng vùng biên (Morph Closing)")

        # B7: Phân vùng màu
        num_labels, labels = cv2.connectedComponents(morph)
        color_segment = np.zeros((morph.shape[0], morph.shape[1], 3), dtype=np.uint8)

        for label in range(1, num_labels):
            mask = labels == label
            color = np.random.randint(0, 255, size=3)
            color_segment[mask] = color

        self.images.append(color_segment)
        self.titles.append("Phân vùng ảnh (Mỗi vùng 1 màu)")

    # ---------- Hiển thị ảnh ----------
    def show_image(self, idx):
        if not self.images:
            return
        img = self.images[idx]
        title = self.titles[idx]
        img_pil = Image.fromarray(cv2.resize(img, (800, 500)))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.img_label.imgtk = imgtk
        self.img_label.configure(image=imgtk)
        self.step_label.config(text=f"🔹 {title}")

    # ---------- Chọn ảnh ----------
    def load_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh",
                                          filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.process_pipeline(path)
            self.index = 0
            self.show_image(self.index)
            self.status_label.config(text=f"✅ Đã chọn ảnh: {path.split('/')[-1]}", fg="green")

    # ---------- Điều hướng ----------
    def next_step(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_image(self.index)

    def prev_step(self):
        if self.index > 0:
            self.index -= 1
            self.show_image(self.index)


# ---------- RUN ----------
if __name__ == "__main__":
    root = Tk()
    app = KirschSegmentationGUI(root)
    root.mainloop()
