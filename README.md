# Spot The Fake Photo (Screen Recapture Detection)

This project provides a production-grade, highly optimized, and lightweight computer vision system to detect whether an input image is a **Real Photo** or a **Photo of a Screen** (screen recapture).

Given one input image, the system outputs a single probability between `0.0` (definitely real) and `1.0` (definitely screen recapture).

---

## Project Structure

```
SpotTheFakePic/
├── Dataset/                   # Dataset containing images
│   ├── realpic/               # Class 0: Real photos
│   └── screen/                # Class 1: Photos of screens (recaptures)
├── src/
│   ├── preprocessing.py       # Image resizing, cropping, color conversions
│   ├── feature_extraction.py  # Handcrafted feature extraction pipelines
│   ├── train.py               # Dataset processing, Scaled SVM training
│   ├── inference.py           # Programmatic Predictor wrapper class
│   └── utils.py               # Serialization & logging utilities
├── models/
│   └── model.pkl              # Saved scaled linear SVM pipeline
├── predict.py                 # Command Line Interface (strict output requirement)
├── requirements.txt           # Minimal library dependencies
├── README.md                  # Project overview & running instructions
└── REPORT.md                  # Project report (approach, accuracy, latency, cost)
```

---

## Installation & Setup

Ensure Python 3.9+ is installed. Clone the repository and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Training the Model
To scan the dataset, extract handcrafted features in parallel, and train/save the Scaled Linear SVM pipeline, run:

```bash
PYTHONPATH=. python3 src/train.py
```

This script will output:
- Feature extraction progress and statistics.
- Performance metrics (Accuracy, Precision, Recall, F1-Score) on a stratified 20% test split.
- Analysis of the Top 10 most influential feature weights.
- Saved pipeline payload inside `models/model.pkl`.

### 2. Predict on an Image (CLI)
To run inference on a single image, use the CLI script `predict.py`. It suppresses all internal logs and libraries' warnings, outputting **exactly** one floating-point probability to `stdout`:

```bash
python3 predict.py Dataset/realpic/IMG_0785.JPG
```
**Example Output:**
```
0.0861
```

```bash
python3 predict.py Dataset/screen/IMG_0913.JPG
```
**Example Output:**
```
0.8969
```

---

## Machine Learning & Feature Engineering Approach

Rather than using a heavy, computationally expensive, and slow Convolutional Neural Network (CNN) that requires high power and cloud GPUs, this project uses **classic Computer Vision** with **handcrafted features** fed into a **Scaled Linear SVM Classifier**.

### Image Representations Used
To ensure high processing speed and preserve high-frequency details, we extract features from three scales:
1. **Resized Image ($512 \times 512$ px)**: Captures global geometry, illumination, edges, and BGR/HSV color distribution histograms.
2. **Center Crop ($256 \times 256$ px) at Original Resolution**: Preserves pristine high-frequency texture details using vectorized Local Binary Patterns (LBP) and 2D FFT spatial frequencies without downsampling artifacts.
3. **Dual-Scale Chromatic analysis**: Computes BGR/HSV histograms and channel ratios on **both** the resized $512 \times 512$ image (global dynamic range) and the original $256 \times 256$ crop (subpixel chromatic fringing and color channel FFTs).

### Feature Extractor Pipeline (474 Dimensions)
- **Laplacian Variance** (2 features): Measured on both resized and crop scales to capture sharpness differences.
- **Edge Density** (2 features): Percentage of Canny edge pixels on both scales.
- **Brightness Statistics** (8 features): Moments (mean, std, skewness, kurtosis) on HSV V channel on both scales.
- **Contrast Statistics** (6 features): RMS contrast, Michelson contrast, and percentile range.
- **Vectorized LBP** (256 features): A custom, fast Local Binary Pattern histogram on the crop.
- **FFT Frequencies** (70 features): Spatial frequency magnitude grid ($8 \times 8$) and concentric ring energies on crop.
- **Gradient Statistics** (20 features): Sobel magnitude stats and 8-bin orientation histograms on both scales.
- **Extended Resized Color Statistics** (46 features): BGR channel histograms (24 features), HSV histograms (16 features), and B/G, R/G, B/R channel ratios (6 features).
- **Extended Crop Color Statistics** (64 features): Crop BGR histograms (24 features), crop HSV histograms (16 features), crop channel ratios (6 features), and color-channel FFT band energies (18 features).

---

## Inference Execution Performance

- **Warm Inference Latency**: **~258.7 ms** (preprocessing + extraction + linear classifier).
- **Model Footprint**: **~332 KB** (making it perfect for mobile phone / on-device compilation).
- **Accuracy**: **$95.83\%$** (with exactly $100\%$ recall on screen recaptures).

---

## Optional Live Webcam Demo

A lightweight, multi-threaded live webcam demonstration server is included.

### How to Run:
1. Start the Flask server:
   ```bash
   PYTHONPATH=. python3 demo.py
   ```
2. Open your web browser and go to:
   `http://127.0.0.1:5000`
   
The web interface displays the live webcam feed with a scanline overlay and dynamically queries prediction statuses (REAL/SCREEN) and probabilities in real-time.
