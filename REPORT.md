# Technical Report: Spot the Fake Photo

This report summarizes the design, evaluation, and operational metrics of the screen recapture detection system.

---

## 1. Approach

The system employs **classic Computer Vision** and **handcrafted feature engineering** combined with a highly optimized, scaled **Linear Support Vector Machine (SVM) pipeline**. This classic approach is chosen over heavy Deep Learning models because it runs extremely fast, has a tiny memory footprint, requires zero GPU acceleration, and generalizes well to edge cases.

To achieve maximum accuracy, features are extracted from three representations:
1. **Resized Image ($512 \times 512$ pixels)**: To extract global geometric and statistical patterns (sharpness, contrast, Canny edge density, Sobel gradient orientations, and BGR/HSV color histograms/ratios).
2. **Original Center Crop ($256 \times 256$ pixels)**: To capture fine, uncompressed texture patterns using **vectorized Local Binary Patterns (LBP)** without resizing artifacts.
3. **Color-Channel FFT Crops**: To extract 2D Fast Fourier Transform band energy statistics (Low, Mid, High) separately for Red, Green, and Blue channels to detect display pixel grid alignments.
4. **Dual-Scale Chromatic analysis**: Computes BGR/HSV histograms and channel ratios on **both** the resized $512 \times 512$ image (global distribution) and the original $256 \times 256$ crop (capturing local display subpixel grids and chromatic fringing).

---

## 2. Model Accuracy

Using a stratified 20% validation split on the provided dataset of 239 images (128 real photos, 111 screen recaptures), the model achieved the following performance metrics:

### Scaled Linear SVM Pipeline (474 features)
- **Accuracy**: $95.83\%$ (46 out of 48 validation images correct)
- **Precision**: $91.67\%$
- **Recall**: $100.00\%$ (Exactly 0 false negatives!)
- **F1-Score**: $95.65\%$
- **Confusion Matrix**:
  ```
  [[24 (True Real)   2 (False Screen)]
   [ 0 (False Real) 22 (True Screen)]]
  ```

---

## 3. Latency Estimates

Latency was evaluated on a standard Apple M-series processor:

- **Warm/In-Process Inference**: **~258.7 ms** per image (optimal for a long-running web service or on-device daemon).
  - **Image loading** ($3\text{--}5\text{MB}$ file): $103.8\text{ ms}$
  - **Feature extraction**: $154.4\text{ ms}$ (highly optimized via crop-based color mapping)
  - **Linear SVM prediction**: $0.5\text{ ms}$
- **Cold Start CLI execution** (`python3 predict.py`): **~2.8 seconds** (primarily driven by python startup and importing heavy libraries like `opencv-python` and `scikit-learn`).

---

## 4. Cost Estimates

- **On-Device Inference**: **$0.00 (Zero cloud cost)**. Because the model is a linear SVM mapping standard features, it executes standard dot products that run on edge device CPUs (e.g. mobile phones) completely free.
- **Cloud CPU Inference (e.g., AWS Lambda)**:
  - Memory footprint: $<180\text{ MB}$.
  - Compute time: $259\text{ ms}$ on a standard 1-vCPU instance.
  - Estimated cost: **~$0.000005 per image** ($5.00 per million images).

---

## 5. Limitations & Future Improvements

### Limitations
- **Extreme Downscaling**: If images are heavily compressed or downscaled by third-party chat apps before processing, fine-grained LBP textures and FFT grid peaks can be degraded, causing false negatives.
- **Surface Reflections**: High ambient reflection on the screen surface can mask moiré textures, making recapture detection harder.

### Future Improvements
1. **Adaptive Salient Cropping**: Instead of a static center crop, locate areas of high contrast/sharpness to isolate screen matrix details dynamically.
2. **gamut Boundaries**: Include chromaticity boundaries to distinguish the restricted color gamut of phone/laptop displays from natural lighting.
3. **On-Device Native Port**: Export the linear SVM weights ($w$ and $b$) and the StandardScaler parameters directly into a native C++/Swift/Kotlin runtime, bypassing Python startup overhead entirely and running under 15ms.
