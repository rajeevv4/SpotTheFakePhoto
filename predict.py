#!/usr/bin/env python3
import sys
import os
import warnings

# Suppress warnings and library messages (e.g., sklearn, cv2, etc.)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Just in case TF or other libs are loaded

# Suppress python logging globally
import logging
logging.disable(logging.CRITICAL)

# Ensure project root is in path to resolve src imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.inference import SpotTheFakePicPredictor

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python predict.py <path_to_image>\n")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        sys.stderr.write(f"Error: File not found at '{image_path}'\n")
        sys.exit(1)
        
    try:
        model_path = os.path.join(project_root, "models", "model.pkl")
        predictor = SpotTheFakePicPredictor(model_path=model_path)
        prob = predictor.predict_probability(image_path)
        
        # Output exactly one floating-point number
        print(f"{prob:.4f}")
        
    except Exception as e:
        sys.stderr.write(f"Error occurred: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
