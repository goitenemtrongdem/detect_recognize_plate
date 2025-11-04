🚗 Automatic Number Plate Recognition (ANPR) – Vietnam Edition

This project implements a real-time Automatic Number Plate Recognition (ANPR) system with a Graphical User Interface (GUI) built using Tkinter.
The system uses OpenCV for image processing, EasyOCR (GPU-accelerated) for optical character recognition, and automatically detects the province of registration from the recognized plate number based on Vietnam's official province code mapping.

🧠 Pipeline Overview

The end-to-end ANPR pipeline consists of five major stages:

1️⃣ GUI Initialization (Tkinter + PIL)

A Tkinter-based GUI provides a user-friendly interface with two main sections:

Left panel: Displays the live camera feed.

Right panel: Displays detected plate information, including:

License plate image preview

Recognized plate number

Province of registration

Date and time of detection

The interface also includes control buttons:

Start → Launches the live camera feed

Detect → Captures and processes the current frame

Exit → Closes the application gracefully

🧩 Technologies used: Tkinter, PIL (Image, ImageTk)

2️⃣ Video Capture & Real-Time Display

When the user presses Start, the app:

Opens the default webcam using cv2.VideoCapture(0)

Reads frames continuously in a loop via update_frame()

Converts each frame from BGR → RGB for correct display colors

Displays the frame in real-time inside the GUI using ImageTk.PhotoImage

🧩 Technologies used: OpenCV, PIL

3️⃣ License Plate Localization

Upon pressing Detect, the system captures a frame and locates the license plate region:

Converts the frame to grayscale

Applies a bilateral filter to reduce noise while preserving edges

gray = cv2.bilateralFilter(gray, 11, 17, 17)


Performs edge detection using the Canny operator:

edged = cv2.Canny(gray, 30, 200)


Finds and sorts contours by area, searching for a 4-sided polygon (the likely plate boundary).

Extracts the plate region (ROI) by masking and cropping the original frame.

📸 The detected plate area is then converted to grayscale and prepared for OCR.

🧩 Technologies used: cv2.findContours, cv2.approxPolyDP, cv2.bitwise_and

4️⃣ Optical Character Recognition (OCR)

The cropped plate image is passed to EasyOCR for text recognition:

result = self.reader.readtext(cropped_image)
text = " ".join([res[1] for res in result]).strip()


The OCR reader is initialized with GPU acceleration (gpu=True), leveraging CUDA-capable GPUs for faster inference.

Extracted text is cleaned and displayed in the GUI.

OCR results may contain misreads (e.g., “O” instead of “0”), which are later normalized during province detection.

🧩 Technologies used: easyocr, numpy

5️⃣ Province Code Extraction & Mapping

After OCR, the recognized text (e.g., "30F-12345") is parsed to extract the first two digits (e.g., "30"), which represent the province code.

Processing steps:

Clean and normalize the OCR output

Replace ambiguous characters:
O/Q → 0, I/L → 1, Z → 2, S → 5, B → 3

Extract only numeric characters.

Lookup the province name from a predefined mapping dictionary (plate_map).

Example:

code = "30" → Province = "Hà Nội"


If no valid match is found, the system returns "Không xác định" (Unknown).

🧩 Technologies used: Python string processing + custom mapping

6️⃣ Display & Logging

Once detection is complete:

The cropped plate image is displayed in the GUI.

Recognized plate number, province, date, and time are populated into their respective fields.

All detections are printed to the console for debug/tracking purposes.

🧩 Technologies used: datetime, Tkinter Entry widgets
[ Start Button ]
       │
       ▼
[ Capture Frame from Camera ]
       │
       ▼
[ Preprocessing ]
(Gray → Bilateral Filter → Edge Detection)
       │
       ▼
[ Contour Analysis ]
→ Find Quadrilateral Region (License Plate)
       │
       ▼
[ OCR with EasyOCR (GPU Accelerated) ]
→ Extract License Plate Text
       │
       ▼
[ Province Detection ]
→ Match First 2 Digits to Vietnam Plate Map
       │
       ▼
[ Display in GUI ]
→ Plate Image, Number, Province, Date & Time
⚡ Performance Notes

EasyOCR supports GPU acceleration (CUDA), significantly improving detection speed.

Bilateral filtering and Canny edge detection are optimized for real-time use.

The pipeline runs smoothly on mid-range GPUs for live recognition tasks.