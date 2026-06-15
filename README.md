# Traffic Guardian Pro 🚦
**Edge-AI Dashcam Surveillance for Wrong-Way Detection**

Traffic Guardian Pro is a lightweight, edge-computing AI system designed to run on standard vehicle dashcams. It uses computer vision to detect and track oncoming vehicles, mathematically calculate their directional vectors to ignore "ego-motion" (camera movement), and extract violator license plates using a high-contrast OCR pipeline.

### Tech Stack
* **Language:** Python
* **Detection:** YOLOv8 (Ultralytics)
* **OCR:** EasyOCR + OpenCV (Bilateral Filtering & CLAHE)
* **GUI:** Tkinter

### Features
* **Stubborn Vector Tracking:** Uses a 12-frame spatial memory to calculate `delta_y` and bounding-box growth rates, completely eliminating false positives from parked cars and road curves.
* **Point-Blank Capture Lock:** Refuses to capture distant, blurry targets. The AI waits until the violator enters a close-range "Strike Zone" before firing the OCR engine.
* **Automated Logging:** Saves high-res evidence crops and logs violations to a local CSV database.

### 📸 Live Demo & Evidence

Here is the AI successfully capturing point-blank evidence of wrong-way violations:

![Wrong Way Detection 1](demo/evidence_1.jpg)

![Wrong Way Detection 2](demo/evidence_2.jpg)

![Wrong Way Detection 2](demo/evidence_3.jpg)

### 📄 Documentation
* [Read the Full End-Term Report Here](docs/Project_Report.pdf)
* [Read the Software Requirements Specification](docs/SRS.pdf)

*Project created for B.Tech Computer Science & Engineering Final Term.*