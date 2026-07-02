import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Union

def extract_laplacian_variance(gray: np.ndarray) -> float:
    """Computes the variance of the Laplacian of the image, which serves as a measure
    of focus and sharpness. Real photos often have different blur/sharpness characteristics
    than photos of screens.

    Args:
        gray: Grayscale image.

    Returns:
        float: Variance of the Laplacian.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))

def extract_edge_density(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> float:
    """Computes the edge density of the image using Canny edge detection.

    Args:
        gray: Grayscale image.
        low_thresh: Lower threshold for Canny.
        high_thresh: Upper threshold for Canny.

    Returns:
        float: Ratio of edge pixels to total pixels.
    """
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return float(np.mean(edges > 0))

def extract_brightness_stats(v_channel: np.ndarray) -> List[float]:
    """Computes statistical moments of the brightness (HSV V channel).

    Args:
        v_channel: Grayscale brightness channel.

    Returns:
        List[float]: [mean, std, skewness, kurtosis] of brightness.
    """
    flat = v_channel.flatten().astype(np.float64)
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    
    # Calculate skewness and kurtosis, handle uniform image case
    if std_val > 1e-5:
        skew_val = float(skew(flat))
        kurt_val = float(kurtosis(flat))
    else:
        skew_val = 0.0
        kurt_val = 0.0
        
    return [mean_val, std_val, skew_val, kurt_val]

def extract_contrast_stats(gray: np.ndarray) -> List[float]:
    """Extracts contrast statistics from the grayscale image.

    Args:
        gray: Grayscale image.

    Returns:
        List[float]: [rms_contrast, michelson_contrast, percentile_contrast]
    """
    flat = gray.flatten().astype(np.float64)
    rms_contrast = float(np.std(flat))
    
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    denom = max_val + min_val
    michelson_contrast = float((max_val - min_val) / denom) if denom > 1e-5 else 0.0
    
    p90 = np.percentile(flat, 90)
    p10 = np.percentile(flat, 10)
    percentile_contrast = float(p90 - p10)
    
    return [rms_contrast, michelson_contrast, percentile_contrast]

def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    """Computes a custom, fast vectorized Local Binary Pattern (LBP) histogram
    over the grayscale image (typically the original-resolution center crop).

    Args:
        gray: Grayscale image.

    Returns:
        np.ndarray: Normalized 256-bin LBP histogram.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros(256, dtype=np.float32)
        
    # Get the center region (excluding 1-pixel border)
    center = gray[1:h-1, 1:w-1]
    
    # Compare center with its 8-neighborhood
    # Shift directions:
    # 0: top-left, 1: top, 2: top-right, 3: right,
    # 4: bottom-right, 5: bottom, 6: bottom-left, 7: left
    neighbors = [
        gray[0:h-2, 0:w-2],
        gray[0:h-2, 1:w-1],
        gray[0:h-2, 2:w],
        gray[1:h-1, 2:w],
        gray[2:h, 2:w],
        gray[2:h, 1:w-1],
        gray[2:h, 0:w-2],
        gray[1:h-1, 0:w-2]
    ]
    
    lbp = np.zeros(center.shape, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        lbp += ((n >= center).astype(np.uint8) << i)
        
    # Compute histogram
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    
    # Normalize histogram
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
        
    return hist

def extract_fft_features(gray: np.ndarray) -> np.ndarray:
    """Computes Fast Fourier Transform (FFT) features to capture high-frequency patterns
    like moiré and screen pixel grids.

    Args:
        gray: Grayscale image.

    Returns:
        np.ndarray: FFT magnitude features (70-dimensional: 8x8 resized magnitude grid
                    plus mean and std in 3 concentric frequency bands).
    """
    # Compute 2D FFT and shift the zero-frequency component to the center
    dft = np.fft.fft2(gray.astype(np.float64))
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1.0)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # 1. Spatial distribution of frequencies (Resize magnitude spectrum to 8x8)
    resized_fft = cv2.resize(magnitude_spectrum, (8, 8), interpolation=cv2.INTER_AREA)
    spatial_features = resized_fft.flatten()
    
    # 2. Concentric frequency band energy
    y_indices, x_indices = np.ogrid[:h, :w]
    r = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
    max_r = np.sqrt(cy**2 + cx**2)
    
    # Define frequency bands (Low, Mid, High)
    low_mask = r < (0.1 * max_r)
    mid_mask = (r >= (0.1 * max_r)) & (r < (0.5 * max_r))
    high_mask = r >= (0.5 * max_r)
    
    band_stats = []
    for mask in [low_mask, mid_mask, high_mask]:
        vals = magnitude_spectrum[mask]
        if len(vals) > 0:
            band_stats.append(float(np.mean(vals)))
            band_stats.append(float(np.std(vals)))
        else:
            band_stats.extend([0.0, 0.0])
            
    return np.concatenate([spatial_features, np.array(band_stats, dtype=np.float32)])

def extract_gradient_stats(gray: np.ndarray) -> np.ndarray:
    """Computes Sobel gradient magnitude and orientation statistics.

    Args:
        gray: Grayscale image.

    Returns:
        np.ndarray: 10-dimensional vector containing magnitude stats (mean, std)
                    and an 8-bin orientation histogram.
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x)  # Range [-pi, pi]
    
    mean_mag = float(np.mean(magnitude))
    std_mag = float(np.std(magnitude))
    
    # Orientation histogram weighted by gradient magnitude
    hist, _ = np.histogram(orientation, bins=8, range=(-np.pi, np.pi), weights=magnitude)
    hist_sum = hist.sum()
    if hist_sum > 1e-5:
        hist /= hist_sum
    else:
        hist = np.zeros(8, dtype=np.float64)
        
    stats = [mean_mag, std_mag] + list(hist)
    return np.array(stats, dtype=np.float32)

def extract_resized_color_features(image_resized: np.ndarray) -> np.ndarray:
    """Extracts BGR histograms, HSV histograms, and channel ratios from the
    resized 512x512 BGR image.

    Args:
        image_resized: Resized BGR image (512x512).

    Returns:
        np.ndarray: 46-dimensional color feature vector.
    """
    hist_features = []
    for chan in range(3):
        hist, _ = np.histogram(image_resized[:, :, chan], bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        hist_features.extend(hist)
        
    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    for chan in (0, 1):
        hist, _ = np.histogram(hsv[:, :, chan], bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        hist_features.extend(hist)
        
    b = image_resized[:, :, 0].astype(np.float32)
    g = image_resized[:, :, 1].astype(np.float32)
    r = image_resized[:, :, 2].astype(np.float32)
    bg_ratio = b / (g + 1e-5)
    rg_ratio = r / (g + 1e-5)
    br_ratio = b / (r + 1e-5)
    
    ratio_stats = [
        float(np.mean(bg_ratio)), float(np.std(bg_ratio)),
        float(np.mean(rg_ratio)), float(np.std(rg_ratio)),
        float(np.mean(br_ratio)), float(np.std(br_ratio))
    ]
    return np.concatenate([np.array(hist_features, dtype=np.float32), 
                           np.array(ratio_stats, dtype=np.float32)])

def extract_crop_color_features(crop: np.ndarray) -> np.ndarray:
    """Extracts BGR histograms, HSV histograms, channel ratios, and color FFT
    band energy statistics from the original-resolution center crop (256x256)
    to capture chromatic grid structures and subpixel aberrations.

    Args:
        crop: Original resolution center crop (256x256 BGR).

    Returns:
        np.ndarray: 64-dimensional feature vector.
    """
    hist_features = []
    for chan in range(3):
        hist, _ = np.histogram(crop[:, :, chan], bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        hist_features.extend(hist)
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    for chan in (0, 1):
        hist, _ = np.histogram(hsv[:, :, chan], bins=8, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
        hist_features.extend(hist)
        
    b = crop[:, :, 0].astype(np.float32)
    g = crop[:, :, 1].astype(np.float32)
    r = crop[:, :, 2].astype(np.float32)
    bg_ratio = b / (g + 1e-5)
    rg_ratio = r / (g + 1e-5)
    br_ratio = b / (r + 1e-5)
    
    ratio_stats = [
        float(np.mean(bg_ratio)), float(np.std(bg_ratio)),
        float(np.mean(rg_ratio)), float(np.std(rg_ratio)),
        float(np.mean(br_ratio)), float(np.std(br_ratio))
    ]
    
    fft_color_stats = []
    crop_size = 256
    for chan in range(3):
        chan_crop = crop[:, :, chan]
        dft = np.fft.fft2(chan_crop.astype(np.float64))
        dft_shift = np.fft.fftshift(dft)
        mag = np.log(np.abs(dft_shift) + 1.0)
        
        cy, cx = crop_size // 2, crop_size // 2
        y_indices, x_indices = np.ogrid[:crop_size, :crop_size]
        dist = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        
        low_mask = dist < (0.1 * max_dist)
        mid_mask = (dist >= (0.1 * max_dist)) & (dist < (0.5 * max_dist))
        high_mask = dist >= (0.5 * max_dist)
        
        for mask in [low_mask, mid_mask, high_mask]:
            vals = mag[mask]
            if len(vals) > 0:
                fft_color_stats.append(float(np.mean(vals)))
                fft_color_stats.append(float(np.std(vals)))
            else:
                fft_color_stats.extend([0.0, 0.0])
                
    return np.concatenate([np.array(hist_features, dtype=np.float32), 
                           np.array(ratio_stats, dtype=np.float32), 
                           np.array(fft_color_stats, dtype=np.float32)])

def extract_features_pipeline(image: np.ndarray) -> np.ndarray:
    """Processes a BGR image, extracts multiscale features, and returns a single
    unified feature vector.

    Args:
        image: Source BGR image.

    Returns:
        np.ndarray: Unified 1D feature vector.
    """
    from src.preprocessing import get_resized_gray, get_center_crop, get_hsv_value_channel
    
    # 1. Generate representations
    gray_resized = get_resized_gray(image, size=512)
    gray_crop = get_center_crop(image, size=256)
    
    # Brightness channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # Resized V channel
    v_resized = cv2.resize(v_channel, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Center crop V channel at original resolution
    h, w = v_channel.shape
    if h >= 256 and w >= 256:
        start_y = (h - 256) // 2
        start_x = (w - 256) // 2
        v_crop = v_channel[start_y:start_y+256, start_x:start_x+256]
    else:
        v_crop = cv2.resize(v_channel, (256, 256), interpolation=cv2.INTER_AREA)
        
    # Get original resolution BGR crop for color features
    h_bgr, w_bgr, _ = image.shape
    if h_bgr >= 256 and w_bgr >= 256:
        start_y = (h_bgr - 256) // 2
        start_x = (w_bgr - 256) // 2
        crop_bgr = image[start_y:start_y+256, start_x:start_x+256]
    else:
        crop_bgr = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
    # 2. Extract features
    features = []
    
    # Laplacian Variance (sharpness)
    features.append(extract_laplacian_variance(gray_resized))
    features.append(extract_laplacian_variance(gray_crop))
    
    # Edge Density
    features.append(extract_edge_density(gray_resized))
    features.append(extract_edge_density(gray_crop))
    
    # Brightness statistics
    features.extend(extract_brightness_stats(v_resized))
    features.extend(extract_brightness_stats(v_crop))
    
    # Contrast statistics
    features.extend(extract_contrast_stats(gray_resized))
    features.extend(extract_contrast_stats(gray_crop))
    
    # Handcrafted texture (LBP) from crop
    features.extend(extract_lbp_features(gray_crop))
    
    # Frequency analysis (FFT) from crop
    features.extend(extract_fft_features(gray_crop))
    
    # Gradient statistics
    features.extend(extract_gradient_stats(gray_resized))
    features.extend(extract_gradient_stats(gray_crop))
    
    # Color features (resized scale)
    image_resized_bgr = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    resized_color_features = extract_resized_color_features(image_resized_bgr)
    
    # Color features (original crop scale)
    crop_color_features = extract_crop_color_features(crop_bgr)
    
    # 3. Clean and convert to float32 array
    feature_vector = np.concatenate([
        np.array(features, dtype=np.float32), 
        resized_color_features, 
        crop_color_features
    ])
    
    # Safeguard against NaN/inf
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector
