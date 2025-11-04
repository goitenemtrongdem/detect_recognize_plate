"""
yolo_plate_ocr_save.py
Mô tả: 
  - Dùng YOLOv8 (ultralytics) với file best.pt để detect biển số.
  - Cắt vùng biển số, tiền xử lý, đọc chữ bằng pytesseract.
  - Ghi text (plate) lên ảnh cạnh bounding box.
  - Lưu ảnh kết quả vào folder output mà không mở GUI.

Sử dụng:
  python yolo_plate_ocr_save.py --input_folder ./input_images --output_folder ./results --model_path ./best.pt
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import pytesseract
import imutils

def preprocess_for_ocr(plate_img):
    """Tiền xử lý crop của biển để tăng độ chính xác OCR"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # tăng tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # lọc nhiễu
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)
    # morphological mở để nối nét chữ bị đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th

def ocr_read_plate(plate_img_bgr, tesseract_config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """
    Nhận crop plate BGR -> trả về chuỗi OCR.
    Config:
      - oem 3 (LSTM), psm 7 (treat image as single text line)
      - whitelist chữ và số (thường cho biển VN/US)
    """
    processed = preprocess_for_ocr(plate_img_bgr)
    # pytesseract yêu cầu image (grayscale or color) -> dùng processed
    text = pytesseract.image_to_string(processed, config=tesseract_config)
    # làm sạch text: bỏ ký tự lạ và khoảng trắng thừa
    text = text.strip()
    # giữ lại A-Z, 0-9, gạch ngang nếu cần
    filtered = ''.join([c for c in text if c.isalnum() or c == '-'])
    return filtered

def draw_plate_label(img, bbox, label_text):
    """
    Ghi label_text vào ảnh bên cạnh bbox.
    bbox = [x1, y1, x2, y2]
    """
    x1,y1,x2,y2 = map(int, bbox)
    # vẽ rectangle
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    # chuẩn bị vị trí text: bên phải bbox, nếu không đủ thì phía trên
    text = label_text if label_text else "UNKNOWN"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = x2 + 10
    ty = y1 + th
    # nếu ra ngoài phải ảnh thì đặt phía trên bbox
    h_img, w_img = img.shape[:2]
    if tx + tw + 10 > w_img:
        tx = x1
        ty = y1 - 10
        if ty - th < 0:
            ty = y2 + th + 10
    # background rectangle for readability
    cv2.rectangle(img, (tx-5, ty-th-5), (tx+tw+5, ty+5), (0,0,0), -1)
    cv2.putText(img, text, (tx, ty), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def crop_rotated_rect(img, box):
    """
    Nếu YOLO trả center/width/height/angle (not typical), hoặc nếu muốn deskew bằng minAreaRect
    box: 4 points contour (x,y)
    => return aligned crop
    """
    rect = cv2.minAreaRect(np.array(box))
    box_pts = cv2.boxPoints(rect).astype("int")
    width = int(rect[1][0])
    height = int(rect[1][1])
    if width == 0 or height == 0:
        return None
    src_pts = box_pts.astype("float32")
    # target coordinates
    dst_pts = np.array([[0, height-1],
                        [0,0],
                        [width-1,0],
                        [width-1,height-1]], dtype="float32")
    # sort source points to match dst (simple approach using imutils)
    box_pts_sorted = imutils.perspective.order_points(box_pts)
    M = cv2.getPerspectiveTransform(box_pts_sorted, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped

def main(args):
    # Tesseract cmd (Windows) - chỉnh nếu cần
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    # load model
    model = YOLO(args.model_path) # ultralytics YOLOv8

    # make output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # process input files
    image_files = []
    if os.path.isdir(args.input_folder):
        for fn in sorted(os.listdir(args.input_folder)):
            if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','tif','tiff')):
                image_files.append(os.path.join(args.input_folder, fn))
    else:
        raise ValueError("input_folder phải là thư mục chứa ảnh")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Không thể đọc ảnh {img_path}, bỏ qua.")
            continue
        orig = img.copy()

        # YOLO predict (return boxes in xyxy)
        # args: imgsz, conf, iou, device can be set by args
        results = model.predict(source=img, imgsz=args.imgsz, conf=args.confidence, iou=args.iou, device=args.device, verbose=False)

        # results is list; for single image use results[0]
        res = results[0]

        # `res.boxes` has boxes; iterate
        # each box: .xyxy, .conf, .cls
        if not hasattr(res, 'boxes') or len(res.boxes) == 0:
            print(f"[INFO] Không tìm thấy biển số trong {os.path.basename(img_path)}")
            # vẫn lưu bản gốc copy nếu bạn muốn
            outname = os.path.join(args.output_folder, os.path.basename(img_path))
            cv2.imwrite(outname, orig)
            continue

        # nếu có nhiều box, ta sẽ xử lý từng box (có thể là nhiều biển trên 1 ảnh)
        plate_texts = []
        for i,box in enumerate(res.boxes):
            # bbox xyxy
            xyxy = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
            x1,y1,x2,y2 = map(int, xyxy)
            # đảm bảo crop trong ảnh
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                plate_text = ""
            else:
                # deskew / hỗ trợ rotate bằng cách kiểm tra ratio
                h,w = crop.shape[:2]
                aspect = w / max(h,1)
                # nếu rất "nằm ngang" thì đọc trực tiếp, nếu nghi ngờ xoay thì thử minAreaRect deskew
                if aspect < 0.3 or aspect > 5:
                    # fallback: try minAreaRect on mask
                    plate_text = ocr_read_plate(crop)
                else:
                    plate_text = ocr_read_plate(crop)

            plate_texts.append(plate_text)
            draw_plate_label(orig, xyxy, plate_text)

        # lưu ảnh kết quả
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(args.output_folder, f"{name}_result{ext}")
        cv2.imwrite(out_path, orig)
        print(f"[SAVED] {out_path}  -> plates: {plate_texts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 license plate detect + OCR -> save images")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder chứa ảnh gốc")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder lưu ảnh kết quả")
    parser.add_argument("--model_path", type=str, default="best.pt", help="Đường dẫn tới file best.pt (YOLOv8)")
    parser.add_argument("--tesseract_cmd", type=str, default="", help="(Optional) Đường dẫn tới tesseract executable (Windows). Ví dụ: C:/Program Files/Tesseract-OCR/tesseract.exe")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO image size")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for YOLO")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="device: 'cpu' or '0' for GPU")
    args = parser.parse_args()
    main(args)
