# Spot The Fake Photo: Screen Recapture Detection

A production-grade, highly optimized, and lightweight Classic Computer Vision system to detect whether an image is a **Real Photo** or a **Photo of a Screen** (screen recapture). 

This system was designed to detect anti-spoofing attempts in mobile apps where users cheat by capturing photos of other devices (laptops, phones) rather than photographing real-world environments directly. Given a single input image, the model outputs a probability value between `0.0` (definitely real) and `1.0` (definitely screen).

---

## 🚀 Key Performance Indicators (KPIs)

* **Classification Accuracy**: **`95.83%`** (46 out of 48 correct predictions on a stratified 20% validation split of the dataset).
* **Security Recall**: **`100.00%`** (Exactly `0` false negatives on screen recaptures; every single screen recapture attempt was successfully blocked).
* **Warm Inference Latency**: **`~258.7 ms`** per image on standard laptop CPU (Apple M-Series).
* **Model Size**: **`~332 KB`** (pruned Standard Scaler + Linear Support Vector Machine pipeline).
* **Cost Per Image**: **`$0.00`** (On-Device Local CPU execution).

---

## 🛠️ Machine Learning & Feature Engineering Approach

Instead of using a resource-intensive Convolutional Neural Network (CNN) that requires cloud GPUs and heavy runtime environments, this project uses **Classic Computer Vision** with **handcrafted features** mapped to a **Scaled Linear SVM Classifier**. 

### 1. Dual-Scale Representation
To preserve high-frequency details while maintaining processing efficiency, features are extracted from two representations:
* **Resized Image ($512 \times 512$ px)**: Extracts global geometric, brightness, and color channel distributions.
* **Center Crop ($256 \times 256$ px) at Original Resolution**: Preserves pristine pixel matrices, display grids, and texture information without downsampling artifacts.

### 2. Handcrafted Features (474 Dimensions)
The feature vector maps the physical differences between camera sensors and electronic displays:
* **Laplacian Variance** (2 features): Checks multiscale focus/blur discrepancies.
* **Edge Density** (2 features): Measures Canny edge pixel densities.
* **Brightness Statistics** (8 features): Computes statistical moments (mean, std, skewness, kurtosis) on HSV V channel.
* **Contrast Statistics** (6 features): Extracts RMS contrast, Michelson contrast, and percentile ranges.
* **Vectorized LBP** (256 features): A custom, fast Local Binary Pattern histogram on the crop to capture micro-textures.
* **Grayscale FFT** (70 features): Computes 2D Fast Fourier Transform magnitude spatial grids ($8 \times 8$) and concentric ring energies to identify screen grid moiré.
* **Gradient Statistics** (20 features): Computes Sobel magnitude stats and 8-bin gradient orientation histograms.
* **Dual-Scale Color Statistics** (110 features): Captures display backlight signatures and color gamut restrictions:
  * *Global BGR/HSV histograms and channel ratios* (46 features) from the resized image.
  * *Local BGR/HSV histograms, ratios, and color-channel FFT concentric bands* (64 features) from the original crop to catch subpixel color aberrations.

---

## 📁 Project Structure

```
SpotTheFakePic/
├── Dataset/                   # Directory containing BGR images
│   ├── realpic/               # Class 0: Real photos
│   └── screen/                # Class 1: Photos of screens (recaptures)
├── src/
│   ├── preprocessing.py       # Image resizing, cropping, color conversions
│   ├── feature_extraction.py  # Handcrafted feature extraction routines
│   ├── train.py               # Dataset processing, Scaled SVM training
│   ├── inference.py           # Programmatic Predictor wrapper class
│   └── utils.py               # Serialization & logging utilities
├── models/
│   └── model.pkl              # Saved Standard Scaler + Linear SVM pipeline
├── templates/
│   └── index.html             # Webcam demo frontend glassmorphic HUD
├── demo.py                    # Multi-threaded web server for live demo
├── predict.py                 # Command Line Interface (strict output format)
├── requirements.txt           # Minimal library dependencies
├── README.md                  # Project overview (this file)
└── REPORT.md                  # Short technical report
```

---

## 💻 How to Run

Ensure Python 3.9+ is installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 1. Train the Classifier
To load the dataset, run parallel feature extraction, train the Scaled Linear SVM pipeline, and serialize the model:
```bash
PYTHONPATH=. python3 src/train.py
```
*Outputs evaluation metrics on the stratified validation set and saves the model inside `models/model.pkl`.*

### 2. Run Single-Image Prediction (CLI)
To check an image, run the command-line interface:
```bash
python3 predict.py Dataset/testimage/IMG_1025.JPG
```
* **Real Photo Output:** `0.0861` (probability close to 0)
* **Screen Recapture Output:** `0.8969` (probability close to 1)

*Note: The CLI is configured to silence all library warnings and logging. It outputs **strictly** one float probability to `stdout`.*

### 3. Start the Optional Live Webcam Demo
To interact with the model in real time using your webcam:
```bash
PYTHONPATH=. python3 demo.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📊 Performance & Operations Report

### 1. Latency Report
* **Warm Inference**: **`~258.7 ms`** per image on standard laptop CPU (Apple M-Series). This covers BGR image loading (~104ms), 474-feature extraction (~154ms), and linear classifier prediction (~0.5ms).
* **Cold Start CLI**: **`~2.8 seconds`**, which is standard Python interpreter startup overhead driven by loading large compiled binaries (`scikit-learn`, `opencv-python`, `scipy`).
* **Mobile Porting Path**: Exporting the linear SVM weights ($w$ and $b$) and scale factors to a native client framework (C++, Swift, or Kotlin) removes Python overhead entirely, bringing latency down to **`< 15 ms`** per frame.

### 2. Cost Per Image Report
* **On-Device (Client-Side)**: **`$0.00`** (Free). Because the model is a lightweight Linear SVM pipeline, it executes standard CPU dot products. It can run locally inside a mobile app or client CPU without incurring cloud compute costs.
* **Cloud CPU Server (e.g., AWS Lambda)**:
  * *Resource limits*: < 180 MB RAM.
  * *Compute speed*: ~259 ms execution.
  * *Cost Estimate*: **`~$0.000005 per image`** ($0.005 per 1,000 images, or **`$5.00 per million images`** processed).
  * *Assumptions*: Calculated on standard AWS Lambda memory allocations (512MB RAM instance running for 259ms) including basic API gateway network routing.
