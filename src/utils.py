import logging
import pickle
from typing import Any

def setup_logger(name: str = "SpotTheFakePic") -> logging.Logger:
    """Sets up a standardized logger for the project.

    Args:
        name: Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def save_model(model: Any, filepath: str) -> None:
    """Serializes and saves a model to disk using pickle.

    Args:
        model: Trained model object to serialize.
        filepath: Target destination path.
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath: str) -> Any:
    """Loads and deserializes a model from disk using pickle.

    Args:
        filepath: Source path of the serialized model.

    Returns:
        Any: The loaded model object.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)
