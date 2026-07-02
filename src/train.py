import os
import glob
import multiprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple, List, Dict, Any, Union

from src.preprocessing import load_image
from src.feature_extraction import extract_features_pipeline
from src.utils import setup_logger, save_model

# Setup logger
logger = setup_logger("Train")

def process_single_image(task: Tuple[str, int]) -> Tuple[Union[np.ndarray, None], Union[int, None], str]:
    """Worker function to process a single image and extract its features.
    Must be a top-level function for multiprocessing pickling.

    Args:
        task: A tuple of (image_path, label).

    Returns:
        Tuple: (feature_vector, label, error_message)
    """
    path, label = task
    try:
        image = load_image(path)
        features = extract_features_pipeline(image)
        return features, label, ""
    except Exception as e:
        return None, None, f"Error processing {path}: {str(e)}"

def load_dataset_and_extract_features(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Scans the dataset directories and extracts feature vectors in parallel.

    Args:
        dataset_dir: Path to the main Dataset directory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features matrix), y (labels array).
    """
    real_dir = os.path.join(dataset_dir, "realpic")
    screen_dir = os.path.join(dataset_dir, "screen")
    
    tasks = []
    
    # Gather real photos (label 0)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in glob.glob(os.path.join(real_dir, ext)):
            tasks.append((path, 0))
            
    # Gather screen captures (label 1)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in glob.glob(os.path.join(screen_dir, ext)):
            tasks.append((path, 1))
            
    if not tasks:
        raise ValueError(f"No images found in {dataset_dir}. Please check your path.")
        
    logger.info(f"Found {len(tasks)} images in total. Extracting features in parallel...")
    
    # Run feature extraction in parallel using Multiprocessing
    num_workers = min(multiprocessing.cpu_count(), len(tasks))
    logger.info(f"Using {num_workers} CPU worker processes.")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_single_image, tasks)
        
    X_list = []
    y_list = []
    
    for feat, label, err in results:
        if err:
            logger.warning(err)
        elif feat is not None:
            X_list.append(feat)
            y_list.append(label)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    logger.info(f"Successfully extracted features for {len(X)} / {len(tasks)} images.")
    logger.info(f"Feature vector dimensionality: {X.shape[1]}")
    logger.info(f"Class distribution: Real (0) = {np.sum(y == 0)}, Screen (1) = {np.sum(y == 1)}")
    
    return X, y

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
    """Evaluates the model and prints performance metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name label for reporting.

    Returns:
        Dict: Dictionary of evaluation metrics.
    """
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    
    logger.info(f"--- {model_name} Evaluation ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

def main():
    dataset_dir = "/Users/rajeev/SpotTheFakePic/Dataset"
    models_dir = "/Users/rajeev/SpotTheFakePic/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data and extract features
    X, y = load_dataset_and_extract_features(dataset_dir)
    
    # 2. Train/test split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Split data: Train size = {len(X_train)}, Test size = {len(X_test)}")
    
    # 3. Train final Linear SVM Pipeline (Standard Scaling + Linear SVM)
    logger.info("Training Scaled Linear Support Vector Machine (SVM) pipeline...")
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=0.2, probability=True, random_state=42)
    )
    svm_pipeline.fit(X_train, y_train)
    
    # Evaluate final model
    eval_metrics = evaluate_model(svm_pipeline, X_test, y_test, "Linear SVM Pipeline")
    
    # Show top feature weights (Linear SVM coefficients)
    scaler = svm_pipeline.named_steps["standardscaler"]
    classifier = svm_pipeline.named_steps["svc"]
    coefficients = classifier.coef_[0]
    
    # Sort features by absolute coefficient magnitude
    abs_weights = np.abs(coefficients)
    top_indices = np.argsort(abs_weights)[::-1][:10]
    
    logger.info("Top 10 Most Influential Features (SVM Weights):")
    for rank, idx in enumerate(top_indices, 1):
        weight = coefficients[idx]
        # Determine feature names heuristically
        if idx < 2:
            name = f"Laplacian_Var_{'Resized' if idx == 0 else 'Crop'}"
        elif idx < 4:
            name = f"Edge_Density_{'Resized' if idx == 2 else 'Crop'}"
        elif idx < 12:
            name = f"Brightness_Stat_{idx - 4}"
        elif idx < 18:
            name = f"Contrast_Stat_{idx - 12}"
        elif idx < 274:
            name = f"LBP_Bin_{idx - 18}"
        elif idx < 344:
            name = f"FFT_Stat_{idx - 274}"
        elif idx < 364:
            name = f"Gradient_Stat_{idx - 344}"
        elif idx < 388:
            name = f"Color_BGR_Hist_{idx - 364}"
        elif idx < 404:
            name = f"Color_HSV_Hist_{idx - 388}"
        elif idx < 410:
            name = f"Color_Ratio_Stat_{idx - 404}"
        else:
            name = f"Color_FFT_Stat_{idx - 410}"
        logger.info(f"{rank}. Index {idx:03d} | {name:<25} | Weight: {weight:+.4f}")
        
    # 4. Save pipeline payload
    model_payload = {
        "pipeline": svm_pipeline,
        "num_features": X.shape[1]
    }
    
    model_path = os.path.join(models_dir, "model.pkl")
    save_model(model_payload, model_path)
    logger.info(f"Saved final trained model pipeline payload to {model_path}")

if __name__ == "__main__":
    main()
