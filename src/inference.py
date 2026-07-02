import os
import numpy as np
from typing import Any

from src.preprocessing import load_image
from src.feature_extraction import extract_features_pipeline
from src.utils import load_model, setup_logger

logger = setup_logger("Inference")

class SpotTheFakePicPredictor:
    """Wrapper class for executing predictions using the trained SpotTheFakePic model."""

    def __init__(self, model_path: str = "/Users/rajeev/SpotTheFakePic/models/model.pkl"):
        """Initializes the predictor by loading the serialized pipeline.

        Args:
            model_path: Path to the model pkl file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train.py first.")
            
        logger.info(f"Loading pipeline from {model_path}...")
        payload = load_model(model_path)
        
        self.pipeline = payload["pipeline"]
        self.num_features = payload.get("num_features", None)
        logger.info("Predictor initialized successfully.")

    def predict_probability(self, image_path: str) -> float:
        """Calculates the probability of the image being a recapture/screen photo.

        Args:
            image_path: Path to the input image file.

        Returns:
            float: Probability in range [0, 1].
                   0 = definitely real, 1 = definitely screen recapture.
        """
        # 1. Preprocess and extract features (428-dimensional vector)
        image = load_image(image_path)
        feature_vector = extract_features_pipeline(image)
        
        # 2. Verify dimensionality compatibility
        if self.num_features is not None and len(feature_vector) != self.num_features:
            raise ValueError(
                f"Feature vector size mismatch. Expected {self.num_features}, but got {len(feature_vector)}."
            )
            
        # 3. Predict probability using the scikit-learn pipeline (applies scaling and SVC)
        features_reshaped = feature_vector.reshape(1, -1)
        probabilities = self.pipeline.predict_proba(features_reshaped)[0]
        
        # Class 1 is screen recapture
        prob_screen = float(probabilities[1])
        return prob_screen
